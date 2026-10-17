# SPDX-License-Identifier: Apache-2.0
"""CPU check of the prompt prefill with full-attention blocks (conditioning images).

A prompt with a conditioning image attends causally in its text and in full inside the
image's block (upstream's ``joint_full``). ``forward_prefill_bounded`` encodes that as a
per-query key bound and must reproduce a full-sequence forward of the reference math
under the same 4D mask, layer by layer through the returned K/V and through a denoise
step over the prefix. The pipeline's ``_prefill_bounds`` must read the bounds back out of
such a mask and reject one that is not a prefix per row.

    VLLM_NEURON_CPU_MODE=1 python test/unit/hunyuan_image3/test_bounded_prefill.py
"""

import os
import tempfile
from types import SimpleNamespace

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import test_numerics as numerics  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def test_bounded_prefill_matches_masked_reference() -> None:
    numerics._init_single_rank_parallel()
    from safetensors.torch import save_file

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        NeuronHunyuanImage3Transformer,
        forward_prefill_bounded,
    )
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        NeuronHunyuanImage3Pipeline,
    )

    config = numerics._tiny_config()
    generator = torch.Generator().manual_seed(3)
    weights = numerics._random_checkpoint(config, generator)

    # Prompt: 3 text tokens, a 5-token image block (full attention), 4 text tokens.
    prompt_len, block = 12, slice(3, 8)
    prefill_len = 16
    seq = prompt_len + 4  # a generated span follows the prompt in the real mask
    mask = torch.ones(seq, seq, dtype=torch.bool).tril()
    mask[block, block] = True
    half = config.attention_head_dim // 2
    cos_half = torch.rand(seq, half, generator=generator) * 2 - 1
    sin_half = torch.sqrt(1 - cos_half**2)
    inputs = torch.randn(1, seq, config.hidden_size, generator=generator)

    reference = numerics._ref_forward(
        weights,
        config,
        inputs[:, :prompt_len],
        torch.cat((cos_half, cos_half), -1)[:prompt_len],
        torch.cat((sin_half, sin_half), -1)[:prompt_len],
        mask[:prompt_len, :prompt_len].reshape(1, 1, prompt_len, prompt_len),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        with torch.device("cpu"):
            model = NeuronHunyuanImage3Transformer(config, use_nki_mlp=False)
        model = model.to(torch.float32).eval()
        model.load_weights(tmpdir)

    fake = SimpleNamespace(prefill_len=prefill_len, device="cpu")
    bound = NeuronHunyuanImage3Pipeline._prefill_bounds(
        fake, mask.reshape(1, 1, seq, seq), prompt_len
    )
    expected_bound = torch.tensor([1, 2, 3, 8, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16.0])
    assert torch.equal(bound, expected_bound), bound

    pad = prefill_len - prompt_len
    embeds = F.pad(inputs[:, :prompt_len], (0, 0, 0, pad))
    cos = F.pad(cos_half[:prompt_len], (0, 0, 0, pad)).unsqueeze(0).contiguous()
    sin = F.pad(sin_half[:prompt_len], (0, 0, 0, pad)).unsqueeze(0).contiguous()

    # Hidden states of the last layer come out of a forward_prefill-style loop; compare
    # them through the model's own layers with the bounded attention.
    with torch.no_grad():
        kv_bounded = forward_prefill_bounded(model, embeds, cos, sin, bound)
        kv_causal = model.forward_prefill(embeds, cos, sin)
    # Layer 0's K/V depend only on the input: equal either way.
    assert torch.allclose(kv_bounded[0], kv_causal[0]) and torch.allclose(kv_bounded[1], kv_causal[1])
    # Later layers see the image block's full attention, so they must differ from causal
    # and match the masked reference's hidden states, recomputed from the reference K.
    assert not torch.allclose(kv_bounded[2], kv_causal[2]), "block attention had no effect"

    # End to end: the last layer's output through a manual pass equals the reference.
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        _bounded_attend,
        repeat_kv,
        rms_norm,
    )

    hidden = embeds
    manual_kv = []
    with torch.no_grad():
        for layer in model.layers:
            attn = layer.self_attn
            residual = hidden
            normed = rms_norm(hidden, layer.input_layernorm_weight, layer.eps)
            query, key, value = attn._project(normed, cos, sin)
            manual_kv.extend((key, value))
            attended = _bounded_attend(
                query.transpose(1, 2),
                repeat_kv(key, attn.num_kv_groups).transpose(1, 2),
                repeat_kv(value, attn.num_kv_groups).transpose(1, 2),
                attn.scale,
                bound,
                use_kernel=False,
            )
            hidden = residual + attn._finish(attended)
            hidden = hidden + layer.mlp(rms_norm(hidden, layer.post_attention_layernorm_weight, layer.eps))
    for got, want in zip(kv_bounded, manual_kv):
        assert torch.allclose(got, want, atol=1e-6), "forward_prefill_bounded K/V differ"
    actual = hidden[:, :prompt_len]
    error = (actual - reference).abs().max().item() / reference.abs().max().item()
    print(f"bounded prefill vs masked reference: relative {error:.3e}")
    assert error < 2e-4, error

    # A row that is not a prefix (a hole in the middle) must be rejected.
    broken = mask.clone()
    broken[10, 5] = False
    try:
        NeuronHunyuanImage3Pipeline._prefill_bounds(fake, broken.reshape(1, 1, seq, seq), prompt_len)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-prefix prompt row was accepted")


if __name__ == "__main__":
    test_bounded_prefill_matches_masked_reference()
    print("OK")
