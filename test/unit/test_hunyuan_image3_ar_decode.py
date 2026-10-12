# SPDX-License-Identifier: Apache-2.0
"""The AR decode step must reproduce a full causal forward, token by token.

``NeuronHunyuanImage3AR`` generates text as a prompt prefill followed by one-token decode
steps against a fixed-size generated-K/V buffer updated by one-hot arithmetic, with the
visible key range and the RoPE row passed in per step. Each of those is a place to be
off by one -- the wrong cache slot, a key visible one step early, the wrong position --
and every one still yields plausible logits. So this compares each step's logits to the
reference math (the same tiny random checkpoint and full-sequence forward as
``test_hunyuan_image3_numerics.py``) on CPU at TP1.

    VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_ar_decode.py
"""

import os
import sys
import tempfile

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_hunyuan_image3_numerics import (  # noqa: E402
    _init_single_rank_parallel,
    _random_checkpoint,
    _ref_forward,
    _ref_rms_norm,
    _tiny_config,
)


def test_decode_steps_match_full_forward() -> None:
    _init_single_rank_parallel()

    from safetensors.torch import save_file

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_ar import (
        NeuronHunyuanImage3AR,
    )
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        NeuronHunyuanImage3Transformer,
    )

    config = _tiny_config()
    generator = torch.Generator().manual_seed(0)
    weights = _random_checkpoint(config, generator)
    vocab = 24
    ln_f = 1.0 + torch.randn(config.hidden_size, generator=generator) * 0.05
    lm_head = torch.randn(vocab, config.hidden_size, generator=generator) * 0.05

    prompt_len = 5  # the prefill covers prompt_len - 1 tokens; the last one is step 0
    steps = 4
    seq = prompt_len + steps - 1
    prefill_len = 7  # right-pads the 4-token prefill by 3
    budget = 6  # generated-K/V slots; two stay unused
    head_dim = config.attention_head_dim
    half = head_dim // 2

    cos_half = torch.rand(seq, half, generator=generator) * 2 - 1
    sin_half = torch.sqrt(1 - cos_half**2)
    cos_full = torch.cat((cos_half, cos_half), dim=-1)
    sin_full = torch.cat((sin_half, sin_half), dim=-1)
    inputs = torch.randn(1, seq, config.hidden_size, generator=generator)

    causal = torch.ones(seq, seq, dtype=torch.bool).tril().reshape(1, 1, seq, seq)
    hidden = _ref_forward(weights, config, inputs, cos_full, sin_full, causal)
    expected = torch.matmul(_ref_rms_norm(hidden, ln_f, config.rms_norm_eps), lm_head.T)[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        with torch.device("cpu"):
            transformer = NeuronHunyuanImage3Transformer(config, use_nki_mlp=False)
        transformer = transformer.to(torch.float32).eval()
        transformer.load_weights(tmpdir)

    # The generator without its pipeline: only what _decode_step reads.
    ar = NeuronHunyuanImage3AR.__new__(NeuronHunyuanImage3AR)
    ar.model = transformer
    ar.eps = config.rms_norm_eps
    ar.vocab_size = vocab
    ar.vocab_shard = vocab
    ar.vocab_start = 0
    ar.tp_size = 1
    ar.tp_group = None
    ar.ln_f_weight = ln_f
    ar.lm_head_t = lm_head.T.contiguous()

    context = prompt_len - 1
    pad = prefill_len - context
    with torch.no_grad():
        prompt_kv = transformer.forward_prefill(
            F.pad(inputs[:, :context], (0, 0, 0, pad)),
            F.pad(cos_half[:context], (0, 0, 0, pad)).unsqueeze(0).contiguous(),
            F.pad(sin_half[:context], (0, 0, 0, pad)).unsqueeze(0).contiguous(),
        )
    kv_heads = transformer.layers[0].self_attn.num_kv_heads
    gen_kv = [torch.zeros(1, budget, kv_heads, head_dim) for _ in range(len(prompt_kv))]
    visible = torch.full((1, 1, 1, prefill_len + budget), -1.0e30)
    visible[..., :context] = 0.0

    worst = 0.0
    for step in range(steps):
        position = context + step
        visible[..., prefill_len + step] = 0.0
        onehot = torch.zeros(1, budget, 1, 1)
        onehot[0, step] = 1
        with torch.no_grad():
            logits, *gen_kv = ar._decode_step(
                inputs[:, position : position + 1],
                cos_half[position : position + 1].unsqueeze(0),
                sin_half[position : position + 1].unsqueeze(0),
                visible.clone(),
                onehot,
                *prompt_kv,
                *gen_kv,
            )
        want = expected[position]
        error = (logits[0] - want).abs().max().item() / want.abs().max().item()
        assert torch.argmax(logits[0]) == torch.argmax(want), f"step {step}: argmax differs"
        worst = max(worst, error)
    print(f"{steps} decode steps match the full causal forward, worst relative {worst:.3e}")
    assert worst < 2e-4, f"decode diverged from the reference: {worst:.3e}"


if __name__ == "__main__":
    test_decode_steps_match_full_forward()
    print("OK")
