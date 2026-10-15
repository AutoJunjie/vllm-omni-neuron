# SPDX-License-Identifier: Apache-2.0
"""CPU check that the row-sharded VAE decode reproduces the whole-image decode.

A small decoder with the real one's structure (five levels of three ResBlocks, DCAE
upsamples, 32-group GroupNorms, a 64-channel latent repeat) decodes the same latents
whole on one process and split two ways by rows across two gloo ranks
(``vae_device.shard_plan`` / ``sharded_stages`` / ``install_sharded_norms``). The halo
plan and the all-reduced GroupNorm statistics make the split exact, so both float32 and
float16 (the default device dtype, with its per-rank power-of-two scaling) must match
the whole decode to rounding.

    python test/unit/test_hunyuan_image3_vae_sharded.py
"""

import copy
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PARTS = 2
LATENT = (1, 16, 1, 32, 8)  # B, C, T, h, w: 32 latent rows split 2 ways (halo 14)


def _vae(only_decoder: bool = True):
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    torch.manual_seed(0)
    vae = AutoencoderKLConv3D.from_config(
        {
            "block_out_channels": [32, 32, 32, 64, 64],
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 16,
            "layers_per_block": 2,
            "ffactor_spatial": 16,
            "ffactor_temporal": 4,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
            "sample_size": 384,
            "sample_tsize": 96,
            "only_decoder": only_decoder,
        }
    ).eval()
    # Random but non-trivial GroupNorm affines, so the statistics actually matter.
    for module in vae.decoder.modules():
        if isinstance(module, torch.nn.GroupNorm):
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.uniform_(-0.2, 0.2)
    return vae.requires_grad_(False)


def _decode(vae, stages, latents, dtype):
    hidden = latents.to(dtype)
    with torch.no_grad():
        for _, stage in stages:
            hidden = stage(hidden)
    return hidden.float()


def _worker(rank: int, port: int, out_dir: str) -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3 import vae_device as vd

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=PARTS)
    latents = torch.randn(LATENT, generator=torch.Generator().manual_seed(1))
    base = _vae()
    for dtype in (torch.float32, torch.float16):
        whole = copy.deepcopy(base).to(dtype)
        vd.install_fp32_norms(whole)
        expected = _decode(whole, vd.decoder_stages(whole), latents, dtype)

        split = copy.deepcopy(base).to(dtype)
        vd.install_fp32_norms(split)
        plan = vd.shard_plan(split, LATENT[-2], PARTS)
        vd.install_sharded_norms(split, plan, rank, PARTS, dist.group.WORLD)
        actual = _decode(split, vd.sharded_stages(split, plan, rank, PARTS, dist.group.WORLD), latents, dtype)

        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        error = (actual - expected).abs().max().item() / expected.abs().max().item()
        print(f"rank {rank} {dtype}: {PARTS}-way row split vs whole decode, relative {error:.3e}", flush=True)
        limit = 1e-5 if dtype == torch.float32 else 2e-2
        assert error < limit, f"rank {rank} {dtype}: sharded decode diverged: {error}"
    if rank == 0:
        print("plan", [(s["name"], s["height"], s["halo"]) for s in plan], flush=True)

    dist.destroy_process_group()


def test_sharded_decode_matches_whole_decode() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, ""), nprocs=PARTS, join=True)


if __name__ == "__main__":
    test_sharded_decode_matches_whole_decode()
    print("OK")
