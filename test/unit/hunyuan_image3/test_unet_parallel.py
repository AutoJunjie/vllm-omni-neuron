# SPDX-License-Identifier: Apache-2.0
"""The TP-split patch embed and final layer must match the replicated ones.

``unet_parallel.py`` splits the ``UNetDown`` / ``UNetUp`` ResBlocks by GroupNorm group
across the TP ranks. Every rank has to produce the replicated path's output exactly: a
channel slice off by one group, a scale/shift row taken from the wrong half, or a gather
reassembled in the wrong order all still yield plausible images. This runs the split on
CPU with one thread per rank and the all-gather replaced by a barrier exchange, in
float32, at the real layout (patch_size 1, 32 groups) with narrower channels.

    VLLM_NEURON_CPU_MODE=1 python test/unit/hunyuan_image3/test_unet_parallel.py
"""

import os
import threading

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import torch  # noqa: E402


class _BarrierGather:
    """``gather`` for ``world`` threads: every rank's tensor, stacked rank-major."""

    def __init__(self, world: int) -> None:
        self.slots = [None] * world
        self.barrier = threading.Barrier(world)

    def for_rank(self, rank: int):
        def gather(tensor: torch.Tensor) -> torch.Tensor:
            self.slots[rank] = tensor
            self.barrier.wait()
            stacked = torch.stack(self.slots)
            self.barrier.wait()  # nobody overwrites a slot until everyone has read it
            return stacked

        return gather


def _modules(latent: int, hidden: int, width: int, emb: int):
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        UNetDown,
        UNetUp,
    )

    torch.manual_seed(0)
    patch_embed = UNetDown(
        patch_size=1, in_channels=latent, emb_channels=emb, hidden_channels=hidden, out_channels=width
    )
    final_layer = UNetUp(
        patch_size=1,
        in_channels=width,
        emb_channels=emb,
        hidden_channels=hidden,
        out_channels=latent,
        out_norm=True,
    )
    # zero_module() leaves the out convolutions at zero; give every weight a value.
    for module in (patch_embed, final_layer):
        for param in module.parameters():
            param.data = torch.randn_like(param) * 0.1
    return patch_embed.eval(), final_layer.eval()


def _run_split(world: int, fn) -> list[torch.Tensor]:
    results = [None] * world
    errors = []
    exchange = _BarrierGather(world)

    def worker(rank: int) -> None:
        try:
            results[rank] = fn(rank, exchange.for_rank(rank))
        except BaseException as error:  # surfaced below; a dead thread would hang the barrier
            errors.append(error)
            exchange.barrier.abort()

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def _check(world: int) -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        _unet_down_tokens,
        _unet_up_image,
    )
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.unet_parallel import (
        UNetShard,
        unet_down_tokens,
        unet_up_image,
    )

    latent, hidden, width, emb, size, batch = 32, 64, 128, 96, 8, 2
    patch_embed, final_layer = _modules(latent, hidden, width, emb)
    latents = torch.randn(batch, latent, size, size)
    t_emb, t_emb2 = torch.randn(batch, emb), torch.randn(batch, emb)
    shards = [UNetShard(patch_embed, final_layer, rank, world) for rank in range(world)]

    with torch.no_grad():
        ref_tokens, token_h, token_w = _unet_down_tokens(patch_embed, latents, t_emb)
        down = _run_split(world, lambda r, g: unet_down_tokens(shards[r], latents, t_emb, g))
        tokens = torch.randn_like(ref_tokens)
        ref_image = _unet_up_image(final_layer, tokens, t_emb2, token_h, token_w)
        up = _run_split(world, lambda r, g: unet_up_image(shards[r], tokens, t_emb2, token_h, token_w, g))

    for rank in range(world):
        got_tokens, got_h, got_w = down[rank]
        assert (got_h, got_w) == (token_h, token_w)
        down_err = ((got_tokens - ref_tokens).norm() / ref_tokens.norm()).item()
        up_err = ((up[rank] - ref_image).norm() / ref_image.norm()).item()
        assert got_tokens.shape == ref_tokens.shape and up[rank].shape == ref_image.shape
        assert down_err < 1e-5, f"world {world} rank {rank}: patch embed off by {down_err:.2e}"
        assert up_err < 1e-5, f"world {world} rank {rank}: final layer off by {up_err:.2e}"
    print(f"world {world}: every rank matches the replicated patch embed and final layer", flush=True)


def test_split_matches_replicated_tp32() -> None:
    _check(32)


def test_split_matches_replicated_tp8() -> None:
    _check(8)


if __name__ == "__main__":
    test_split_matches_replicated_tp32()
    test_split_matches_replicated_tp8()
