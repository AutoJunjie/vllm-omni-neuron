# SPDX-License-Identifier: Apache-2.0
"""CPU check that the CFG-replica denoise step matches the TP-only one.

Two gloo ranks build the tiny backbone from one random checkpoint, once per layout,
with the groups from vLLM-Omni's ``initialize_model_parallel`` as the engine builds them
from ``parallel_config``: TP2 (the reference, both CFG branches batched) and TP1 x two
CFG replicas (``cfg_parallel_size: 2``). With replicas each rank prefills and denoises
only its own CFG branch, with the pipeline's ``_key_biases``, and ``cfg_gather_batch``
returns both branches -- which must reproduce the TP2 step on every token of both rows.

    VLLM_NEURON_CPU_MODE=1 python test/unit/hunyuan_image3/test_cfg_parallel.py
"""

import os
import socket
import tempfile
from types import SimpleNamespace

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import test_numerics as numerics  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn.functional as F  # noqa: E402

WORLD = 2


def _init(rank: int, port: int, tp: int, cfg: int, ring: int) -> None:
    import vllm.distributed.parallel_state as vllm_ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    import vllm_omni_neuron.bootstrap  # noqa: F401

    # vLLM's same-node probe touches the accelerator, which CPU mode does not register;
    # both ranks are local by construction.
    vllm_ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * WORLD
    # Initialise the world group first: vLLM-Omni's own path picks a device by rank
    # modulo the device count, which is 0 in CPU mode; given a live world it skips that.
    method = f"tcp://127.0.0.1:{port}"
    torch.distributed.init_process_group("gloo", init_method=method, world_size=WORLD, rank=rank)
    init_distributed_environment(
        world_size=WORLD, rank=rank, distributed_init_method=method, local_rank=rank, backend="gloo"
    )
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    _init._context = context
    initialize_model_parallel(
        tensor_parallel_size=tp, cfg_parallel_size=cfg, ring_degree=ring, backend="gloo"
    )


def _worker(rank: int, port: int, checkpoint_dir: str, layout: tuple, reference: str) -> None:
    tp, cfg = layout
    _init(rank, port, tp, cfg, 1)
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        NeuronHunyuanImage3Transformer,
        cfg_gather_batch,
    )
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        NeuronHunyuanImage3Pipeline,
    )

    config = numerics._tiny_config()
    generator = torch.Generator().manual_seed(1)
    prompt_len, image_tokens, prefill_len, bsz = 5, 4, 7, 2
    image_len = 1 + image_tokens
    seq = prompt_len + image_len + 3
    half = config.attention_head_dim // 2

    mask = torch.ones(seq, seq, dtype=torch.bool).tril()
    image_slice = slice(prompt_len + 1, prompt_len + image_len)
    mask[image_slice, image_slice] = True
    attention_mask = mask.reshape(1, 1, seq, seq).expand(bsz, 1, seq, seq)
    cos_half = torch.rand(seq, half, generator=generator) * 2 - 1
    sin_half = torch.sqrt(1 - cos_half**2)
    # Two CFG branches: different prompts, the same step inputs (shared latents).
    inputs = torch.randn(bsz, seq, config.hidden_size, generator=generator)
    inputs[1, prompt_len:] = inputs[0, prompt_len:]

    pad = prefill_len - prompt_len
    prompt_embeds = F.pad(inputs[:, :prompt_len], (0, 0, 0, pad))
    prompt_cos = F.pad(cos_half[:prompt_len], (0, 0, 0, pad)).expand(bsz, prefill_len, half)
    prompt_sin = F.pad(sin_half[:prompt_len], (0, 0, 0, pad)).expand(bsz, prefill_len, half)
    step = slice(prompt_len, prompt_len + image_len)
    step_hidden = inputs[:, step]
    step_cos = cos_half[step].expand(bsz, image_len, half).contiguous()
    step_sin = sin_half[step].expand(bsz, image_len, half).contiguous()

    with torch.device("cpu"):
        model = NeuronHunyuanImage3Transformer(config, use_nki_mlp=False)
    model = model.to(torch.float32).eval()
    model.load_weights(checkpoint_dir)
    assert (model.tp_size, model.cfg_size) == (tp, cfg)
    fake_pipeline = SimpleNamespace(prefill_len=prefill_len, model=model, device="cpu")
    biases = NeuronHunyuanImage3Pipeline._key_biases(
        fake_pipeline, attention_mask, prompt_len, image_len
    )
    rows = slice(model.cfg_rank, model.cfg_rank + 1) if cfg > 1 else slice(None)
    with torch.no_grad():
        prompt_kv = model.forward_prefill(
            prompt_embeds[rows].contiguous(),
            prompt_cos[rows].contiguous(),
            prompt_sin[rows].contiguous(),
        )
        out = model.forward_denoise(step_hidden[rows], step_cos[rows], step_sin[rows], biases, *prompt_kv)
        if cfg > 1:
            out = cfg_gather_batch(out, cfg, model.cfg_group)

    if cfg == 1:
        if rank == 0:
            torch.save(out, reference)
        return
    expected = torch.load(reference)
    assert out.shape == expected.shape, (out.shape, expected.shape)
    error = (out - expected).abs().max().item() / expected.abs().max().item()
    print(f"rank {rank}: TP1 x CFG2 vs TP2 relative error {error:.3e}")
    assert error < 1e-5, f"rank {rank}: CFG-parallel step diverged: {error}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_cfg_parallel_matches_tensor_parallel() -> None:
    from safetensors.torch import save_file

    weights = numerics._random_checkpoint(numerics._tiny_config(), torch.Generator().manual_seed(0))
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        reference = os.path.join(tmpdir, "tp2.pt")
        # (tensor_parallel_size, cfg_parallel_size): the reference first.
        for layout in ((WORLD, 1), (1, WORLD)):
            mp.spawn(_worker, args=(_free_port(), tmpdir, layout, reference), nprocs=WORLD, join=True)


if __name__ == "__main__":
    test_cfg_parallel_matches_tensor_parallel()
    print("OK")
