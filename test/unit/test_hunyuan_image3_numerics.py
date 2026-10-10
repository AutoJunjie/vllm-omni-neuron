# SPDX-License-Identifier: Apache-2.0
"""CPU numerics check for the Neuron HunyuanImage-3.0 backbone.

Builds a tiny random checkpoint in the real HunyuanImage-3.0 key layout and asserts
that the plugin's prefill + denoise split reproduces a single full-sequence forward of
the reference math (the ``modeling_hunyuan_image_3.py`` attention / MoE / RoPE), which
covers at once:

* the fused ``qkv_proj`` loader (rows grouped per KV head: ``g`` query heads, then that
  group's K and V heads);
* the ``gate_and_up_proj`` chunk order (``down(chunk0 * silu(chunk1))``);
* 2D RoPE applied *before* the per-head QK RMSNorm;
* the float32 router -> top-k -> renormalise path expressed as a threshold;
* the claim that splitting the first step into a causal prompt prefill plus a
  full-attention denoise step is exact for HunyuanImage3's generation mask, including
  the prompt right-padding masked by ``key_bias``.

Runs on CPU with ``VLLM_NEURON_CPU_MODE=1`` at ``tensor_parallel_size=1``, so the NKI
kernels take their torch fallbacks. Run it directly or under pytest:

    VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_numerics.py
"""

import os
import tempfile
from types import SimpleNamespace

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=16,
        rms_norm_eps=1e-5,
        use_qk_norm=True,
        num_experts=4,
        moe_topk=2,
        norm_topk_prob=True,
        moe_intermediate_size=32,
        intermediate_size=32,
        num_shared_expert=1,
        use_mixed_mlp_moe=1,
        moe_layer_num_skipped=0,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10000.0,
    )


def _random_checkpoint(config, generator) -> dict[str, torch.Tensor]:
    """Random weights under the real checkpoint key names and shapes."""

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.05

    hidden = config.hidden_size
    head_dim = config.attention_head_dim
    kv_heads = config.num_key_value_heads
    groups = config.num_attention_heads // kv_heads
    inter = config.moe_intermediate_size

    weights: dict[str, torch.Tensor] = {}
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = 1.0 + randn(hidden)
        weights[f"{prefix}.post_attention_layernorm.weight"] = 1.0 + randn(hidden)
        weights[f"{prefix}.self_attn.qkv_proj.weight"] = randn(
            kv_heads * (groups + 2) * head_dim, hidden
        )
        weights[f"{prefix}.self_attn.o_proj.weight"] = randn(hidden, hidden)
        weights[f"{prefix}.self_attn.query_layernorm.weight"] = 1.0 + randn(head_dim)
        weights[f"{prefix}.self_attn.key_layernorm.weight"] = 1.0 + randn(head_dim)
        weights[f"{prefix}.mlp.gate.wg.weight"] = randn(config.num_experts, hidden)
        weights[f"{prefix}.mlp.shared_mlp.gate_and_up_proj.weight"] = randn(
            2 * config.intermediate_size, hidden
        )
        weights[f"{prefix}.mlp.shared_mlp.down_proj.weight"] = randn(
            hidden, config.intermediate_size
        )
        for expert in range(config.num_experts):
            weights[f"{prefix}.mlp.experts.{expert}.gate_and_up_proj.weight"] = randn(
                2 * inter, hidden
            )
            weights[f"{prefix}.mlp.experts.{expert}.down_proj.weight"] = randn(hidden, inter)
    return weights


# ---------------------------------------------------------------------------
# Reference math, transcribed from modeling_hunyuan_image_3.py
# ---------------------------------------------------------------------------


def _ref_rms_norm(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf


def _ref_rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _ref_apply_rope(q, k, cos, sin):
    """``cos``/``sin`` are full-width ``[S, D]`` (the reference's ``.repeat(1, 2)``)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (
        q * cos + _ref_rotate_half(q) * sin,
        k * cos + _ref_rotate_half(k) * sin,
    )


def _ref_mlp(x, fused, down):
    chunk0, chunk1 = torch.matmul(x, fused.T).chunk(2, dim=-1)
    return torch.matmul(chunk0 * F.silu(chunk1), down.T)


def _ref_moe(x, weights, prefix, config):
    logits = torch.matmul(x.float(), weights[f"{prefix}.mlp.gate.wg.weight"].float().T)
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_i = torch.topk(probs, config.moe_topk, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    out = torch.zeros_like(x)
    for slot in range(config.moe_topk):
        for expert in range(config.num_experts):
            selected = topk_i[:, slot] == expert
            if not bool(selected.any()):
                continue
            expert_out = _ref_mlp(
                x[selected],
                weights[f"{prefix}.mlp.experts.{expert}.gate_and_up_proj.weight"],
                weights[f"{prefix}.mlp.experts.{expert}.down_proj.weight"],
            )
            out[selected] += expert_out * topk_w[selected, slot].unsqueeze(-1)
    shared = _ref_mlp(
        x,
        weights[f"{prefix}.mlp.shared_mlp.gate_and_up_proj.weight"],
        weights[f"{prefix}.mlp.shared_mlp.down_proj.weight"],
    )
    return out + shared


def _ref_forward(weights, config, inputs_embeds, cos_full, sin_full, attn_mask):
    hidden = inputs_embeds
    head_dim = config.attention_head_dim
    kv_heads = config.num_key_value_heads
    heads = config.num_attention_heads
    groups = heads // kv_heads
    bsz, seq, _ = hidden.shape

    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        residual = hidden
        x = _ref_rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)

        qkv = torch.matmul(x, weights[f"{prefix}.self_attn.qkv_proj.weight"].T)
        qkv = qkv.reshape(bsz, seq, kv_heads, groups + 2, head_dim)
        q, k, v = torch.split(qkv, [groups, 1, 1], dim=3)
        q = q.reshape(bsz, seq, heads, head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq, kv_heads, head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq, kv_heads, head_dim).transpose(1, 2)

        q, k = _ref_apply_rope(q, k, cos_full, sin_full)
        q = _ref_rms_norm(q, weights[f"{prefix}.self_attn.query_layernorm.weight"], config.rms_norm_eps)
        k = _ref_rms_norm(k, weights[f"{prefix}.self_attn.key_layernorm.weight"], config.rms_norm_eps)

        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(bsz, seq, heads * head_dim)
        hidden = residual + torch.matmul(attn, weights[f"{prefix}.self_attn.o_proj.weight"].T)

        residual = hidden
        x = _ref_rms_norm(
            hidden, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
        )
        moe = torch.stack(
            [_ref_moe(x[b], weights, prefix, config) for b in range(bsz)],
            dim=0,
        )
        hidden = residual + moe
    return hidden


# ---------------------------------------------------------------------------


def _init_single_rank_parallel() -> None:
    """Bring up a one-rank TP group so the backbone's size queries resolve."""
    import vllm_omni_neuron.bootstrap  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    # initialize_model_parallel reads the ambient vLLM config; keep the context open
    # for the rest of the process so the groups stay valid.
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    _init_single_rank_parallel._context = context  # keep a reference alive
    initialize_model_parallel(tensor_model_parallel_size=1)


def test_prefill_denoise_split_matches_reference() -> None:
    _init_single_rank_parallel()

    from safetensors.torch import save_file

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        NeuronHunyuanImage3Transformer,
    )

    config = _tiny_config()
    generator = torch.Generator().manual_seed(0)
    weights = _random_checkpoint(config, generator)

    prompt_len = 5
    image_tokens = 4
    image_len = 1 + image_tokens  # timestep token + image tokens
    # The real template emits more tokens after the last <img>: <eoi>, then the
    # answer/bot suffixes. They are NOT part of a denoise step (upstream's steady-state
    # step drops them too), so the sequence here is longer than the step's span and the
    # plugin must bound every slice by prompt_len + image_len rather than by `seq`.
    trailing = 3
    seq = prompt_len + image_len + trailing
    prefill_len = 7  # right-pads the prompt by 2
    bsz = 2
    hidden_size = config.hidden_size
    head_dim = config.attention_head_dim

    # Generation mask: causal, with a full-attention block over the image tokens only
    # (the timestep token at `prompt_len` stays causal — the stricter of the two
    # conventions, so the two-bias path is genuinely exercised).
    mask = torch.ones(seq, seq, dtype=torch.bool).tril()
    image_slice = slice(prompt_len + 1, prompt_len + image_len)
    mask[image_slice, image_slice] = True

    half = head_dim // 2
    cos_half = torch.rand(seq, half, generator=generator) * 2 - 1
    sin_half = torch.sqrt(1 - cos_half**2)
    cos_full = torch.cat((cos_half, cos_half), dim=-1)
    sin_full = torch.cat((sin_half, sin_half), dim=-1)

    inputs = torch.randn(bsz, seq, hidden_size, generator=generator)

    reference = _ref_forward(
        weights, config, inputs, cos_full, sin_full, mask.reshape(1, 1, seq, seq)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        with torch.device("cpu"):
            transformer = NeuronHunyuanImage3Transformer(config, use_nki_mlp=False)
        transformer = transformer.to(torch.float32).eval()
        transformer.load_weights(tmpdir)

    # --- prefill over the right-padded prompt ---
    pad = prefill_len - prompt_len
    prompt_embeds = F.pad(inputs[:, :prompt_len], (0, 0, 0, pad))
    prompt_cos = F.pad(cos_half[:prompt_len], (0, 0, 0, pad)).unsqueeze(0).expand(
        bsz, prefill_len, half
    )
    prompt_sin = F.pad(sin_half[:prompt_len], (0, 0, 0, pad)).unsqueeze(0).expand(
        bsz, prefill_len, half
    )
    with torch.no_grad():
        prompt_kv = transformer.forward_prefill(
            prompt_embeds, prompt_cos.contiguous(), prompt_sin.contiguous()
        )

    # --- denoise step over [timestep token] + image tokens ---
    neg = -1.0e30
    total_keys = prefill_len + image_len

    def build_bias(row: torch.Tensor) -> torch.Tensor:
        bias = torch.full((1, 1, 1, total_keys), neg)
        bias[..., :prompt_len] = 0.0
        bias[0, 0, 0, prefill_len:] = torch.where(
            row[prompt_len : prompt_len + image_len], torch.zeros(()), torch.full((), neg)
        )
        return bias

    timestep_bias = build_bias(mask[prompt_len])
    image_bias = build_bias(mask[prompt_len + 1])

    step = slice(prompt_len, prompt_len + image_len)
    step_cos = cos_half[step].unsqueeze(0).expand(bsz, image_len, half).contiguous()
    step_sin = sin_half[step].unsqueeze(0).expand(bsz, image_len, half).contiguous()
    with torch.no_grad():
        actual = transformer.forward_denoise(
            inputs[:, step],
            step_cos,
            step_sin,
            (timestep_bias, image_bias),
            *prompt_kv,
        )

    expected = reference[:, step]
    error = (actual - expected).abs().max().item()
    scale = expected.abs().max().item()
    print(f"max abs error {error:.3e} (signal {scale:.3e}, relative {error / scale:.3e})")
    assert error / scale < 2e-4, f"prefill+denoise split diverged: {error / scale}"


if __name__ == "__main__":
    test_prefill_denoise_split_matches_reference()
    print("OK")
