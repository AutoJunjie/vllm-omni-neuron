# SPDX-License-Identifier: Apache-2.0
"""CPU check that the CFG-replica / context-parallel denoise step matches the TP-only one.

Two gloo ranks build the tiny backbone from one random checkpoint, once per layout,
with the groups from vLLM-Omni's ``initialize_model_parallel`` as the engine builds them
from ``parallel_config``: TP2 (the reference), TP1 x two CFG replicas
(``cfg_parallel_size: 2``) and TP1 x CP2 (``ring_degree: 2``). On the second axis each
rank holds half of the (padded) denoise tokens and the attention all-gathers K/V across
the pair; the step uses the pipeline's own pieces -- ``_key_biases`` (padding columns
masked, the timestep row only on rank 0 of the pair), ``cp_local_tokens`` for the hidden
states and RoPE, and ``cp_gather_tokens`` for the output -- and must reproduce the TP2
step on every live token. With CFG replicas, each rank also runs one batch row whole and
gathers the batch (CFG parallel), and the AR decode step, where the pair splits its
shared shard's heads, experts and LM-head rows and reduces over the world, must
reproduce TP2's logits.

    VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_context_parallel.py
"""

import os
import socket
import tempfile
from types import SimpleNamespace

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import test_hunyuan_image3_numerics as numerics  # noqa: E402
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
    tp, cfg, ring = layout
    _init(rank, port, tp, cfg, ring)
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_ar import (
        NeuronHunyuanImage3AR,
    )
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        NeuronHunyuanImage3Transformer,
        cp_gather_batch,
        cp_gather_tokens,
        cp_local_tokens,
    )
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        NeuronHunyuanImage3Pipeline,
    )

    config = numerics._tiny_config()
    generator = torch.Generator().manual_seed(1)
    prompt_len, image_tokens, prefill_len, bsz = 5, 4, 7, 2
    image_len = 1 + image_tokens  # odd, so CP2 pads one token
    seq = prompt_len + image_len + 3
    half = config.attention_head_dim // 2

    mask = torch.ones(seq, seq, dtype=torch.bool).tril()
    image_slice = slice(prompt_len + 1, prompt_len + image_len)
    mask[image_slice, image_slice] = True
    attention_mask = mask.reshape(1, 1, seq, seq).expand(bsz, 1, seq, seq)
    cos_half = torch.rand(seq, half, generator=generator) * 2 - 1
    sin_half = torch.sqrt(1 - cos_half**2)
    inputs = torch.randn(bsz, seq, config.hidden_size, generator=generator)

    pad = prefill_len - prompt_len
    prompt_embeds = F.pad(inputs[:, :prompt_len], (0, 0, 0, pad))
    prompt_cos = F.pad(cos_half[:prompt_len], (0, 0, 0, pad)).expand(bsz, prefill_len, half)
    prompt_sin = F.pad(sin_half[:prompt_len], (0, 0, 0, pad)).expand(bsz, prefill_len, half)
    step = slice(prompt_len, prompt_len + image_len)
    step_hidden = inputs[:, step]
    step_cos = cos_half[step].expand(bsz, image_len, half).contiguous()
    step_sin = sin_half[step].expand(bsz, image_len, half).contiguous()

    vocab = 24
    ln_f = 1.0 + torch.randn(config.hidden_size, generator=generator) * 0.05
    lm_head = torch.randn(vocab, config.hidden_size, generator=generator) * 0.05
    budget = 3
    visible = torch.full((1, 1, 1, prefill_len + budget), -1.0e30)
    visible[..., : prompt_len + 1] = 0.0
    visible[..., prefill_len : prefill_len + 2] = 0.0
    onehot = torch.zeros(1, budget, 1, 1)
    onehot[0, 1] = 1

    outputs, logits = {}, {}
    cp = cfg * ring
    for _ in (None,):
        with torch.device("cpu"):
            model = NeuronHunyuanImage3Transformer(config, use_nki_mlp=False)
        model = model.to(torch.float32).eval()
        model.load_weights(checkpoint_dir)
        assert (model.tp_size, model.cp_size, model.cfg_replicas) == (tp, cp, cfg > 1)
        fake_pipeline = SimpleNamespace(prefill_len=prefill_len, model=model, device="cpu")
        biases = NeuronHunyuanImage3Pipeline._key_biases(
            fake_pipeline, attention_mask, prompt_len, image_len
        )
        hidden, cos, sin = step_hidden, step_cos, step_sin
        if cp > 1:
            hidden, cos, sin = (cp_local_tokens(t, cp, model.cp_rank) for t in (hidden, cos, sin))
        with torch.no_grad():
            prompt_kv = model.forward_prefill(
                prompt_embeds.contiguous(), prompt_cos.contiguous(), prompt_sin.contiguous()
            )
            out = model.forward_denoise(hidden, cos, sin, biases, *prompt_kv)
            if cp > 1:
                out = cp_gather_tokens(out, cp, model.cp_group)
        outputs["step"] = out[:, :image_len]

        if model.cfg_replicas:
            # CFG parallel: this rank's batch row through the whole step, no K/V exchange.
            row = slice(model.cp_rank, model.cp_rank + 1)
            whole = NeuronHunyuanImage3Pipeline._key_biases(
                fake_pipeline, attention_mask, prompt_len, image_len, split_tokens=False
            )
            with torch.no_grad():
                row_kv = model.forward_prefill(
                    prompt_embeds[row].contiguous(),
                    prompt_cos[row].contiguous(),
                    prompt_sin[row].contiguous(),
                )
                out = model.forward_denoise(
                    step_hidden[row], step_cos[row], step_sin[row], whole, *row_kv,
                    split_tokens=False,
                )
            outputs["cfg"] = cp_gather_batch(out, cp, model.cp_group)

        # AR decode of one token at slot 1 of the generated cache, as __init__ sets it up.
        ar = NeuronHunyuanImage3AR.__new__(NeuronHunyuanImage3AR)
        ar.model, ar.eps, ar.vocab_size = model, config.rms_norm_eps, vocab
        ar.tp_size, ar.tp_group = model.tp_size, model.tp_group
        ar.part, ar.parts = model.cp_rank, model.cp_size
        ar.reduce_group = model.world_group if cp > 1 else model.tp_group
        shard = vocab // model.tp_size
        part = shard // cp
        ar.logits_start = model.tp_rank * shard + model.cp_rank * part
        ar.ln_f_weight = ln_f
        ar.lm_head_t = lm_head[ar.logits_start : ar.logits_start + part].T.contiguous()
        # Earlier generated K/V: the same random cache, this rank's KV heads of it.
        kv_heads = model.layers[0].self_attn.num_kv_heads
        heads = slice(model.tp_rank * kv_heads, (model.tp_rank + 1) * kv_heads)
        cache = torch.Generator().manual_seed(7)
        gen_kv = [
            torch.randn(
                1, budget, config.num_key_value_heads, config.attention_head_dim, generator=cache
            )[:, :, heads]
            for _ in range(2 * config.num_hidden_layers)
        ]
        with torch.no_grad():
            prompt_kv = model.forward_prefill(
                prompt_embeds.contiguous(), prompt_cos.contiguous(), prompt_sin.contiguous()
            )
            step_logits, *_ = ar._decode_step(
                inputs[:, prompt_len : prompt_len + 1],
                cos_half[prompt_len : prompt_len + 1].unsqueeze(0),
                sin_half[prompt_len : prompt_len + 1].unsqueeze(0),
                visible,
                onehot,
                *prompt_kv,
                *gen_kv,
            )
        logits["ar"] = step_logits

    if cp == 1:
        # tp_sequence_parallel on the same TP2 groups: the residual stream, norms and
        # router on half the tokens each, gathered into the attention and the experts.
        with torch.device("cpu"):
            sp_model = NeuronHunyuanImage3Transformer(config, use_nki_mlp=False, tp_sequence_parallel=True)
        sp_model = sp_model.to(torch.float32).eval()
        sp_model.load_weights(checkpoint_dir)
        assert sp_model.sp_enabled
        with torch.no_grad():
            sp_kv = sp_model.forward_prefill(
                prompt_embeds.contiguous(), prompt_cos.contiguous(), prompt_sin.contiguous()
            )
            sp_out = sp_model.forward_denoise(sp_model.sp_shard(step_hidden), step_cos, step_sin, biases, *sp_kv)
            sp_out = sp_model.sp_unshard(sp_out, image_len)
        scale = outputs["step"].abs().max().item()
        error = (sp_out - outputs["step"]).abs().max().item() / scale
        print(f"rank {rank}: TP2 sequence parallel vs TP2 relative error {error:.3e}")
        assert error < 1e-5, f"rank {rank}: sequence-parallel step diverged: {error}"
        if rank == 0:
            torch.save({"step": outputs["step"], "ar": logits["ar"]}, reference)
        return
    expected = torch.load(reference)
    name = f"TP{tp} x {'CFG' if cfg > 1 else 'CP'}{cp}"
    scale = expected["step"].abs().max().item()
    error = (outputs["step"] - expected["step"]).abs().max().item() / scale
    print(f"rank {rank}: {name} token split vs TP2 relative error {error:.3e}")
    assert error < 1e-5, f"rank {rank}: {name} token-split step diverged: {error}"
    if model.cfg_replicas:
        error = (outputs["cfg"] - expected["step"]).abs().max().item() / scale
        print(f"rank {rank}: {name} CFG parallel vs TP2 relative error {error:.3e}")
        assert error < 1e-5, f"rank {rank}: CFG-parallel step diverged: {error}"
    error = (logits["ar"] - expected["ar"]).abs().max().item() / expected["ar"].abs().max().item()
    print(f"rank {rank}: {name} split AR decode vs TP2 logits relative error {error:.3e}")
    assert error < 1e-5, f"rank {rank}: split AR decode diverged: {error}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_context_parallel_matches_tensor_parallel() -> None:
    from safetensors.torch import save_file

    weights = numerics._random_checkpoint(numerics._tiny_config(), torch.Generator().manual_seed(0))
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        reference = os.path.join(tmpdir, "tp2.pt")
        # (tensor_parallel_size, cfg_parallel_size, ring_degree): the reference first.
        for layout in ((WORLD, 1, 1), (1, WORLD, 1), (1, 1, WORLD)):
            mp.spawn(_worker, args=(_free_port(), tmpdir, layout, reference), nprocs=WORLD, join=True)


if __name__ == "__main__":
    test_context_parallel_matches_tensor_parallel()
    print("OK")
