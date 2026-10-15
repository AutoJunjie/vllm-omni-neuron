# SPDX-License-Identifier: Apache-2.0
"""The staged device decode must compute what ``AutoencoderKLConv3D.decode`` computes.

The pipeline's ``vae_device: neuron`` path replaces one ``vae.decode`` call with seven
separately compiled stages and a float32-statistics GroupNorm. A dropped block or a
mis-ordered stage still decodes to a plausible image, so this checks both pieces against
the reference modules on CPU, in float32, on a small random VAE of the same architecture.

    VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_vae_device.py
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import torch
import torch.nn as nn

from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (
    decoder_stages,
    install_fp32_attention,
    install_fp32_norms,
)

# Same shape of config as the checkpoint's (5 levels, 16x spatial, 4x temporal), with
# narrow channels so the decode runs in a second on CPU.
_SMALL_VAE = {
    "block_out_channels": [64, 64, 32, 32, 32],
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 1,
    "ffactor_spatial": 16,
    "ffactor_temporal": 4,
    "sample_size": 64,
    "sample_tsize": 16,
    "downsample_match_channel": True,
    "upsample_match_channel": True,
    "only_decoder": True,
}


def _small_vae():
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    torch.manual_seed(0)
    vae = AutoencoderKLConv3D.from_config(_SMALL_VAE).eval()
    vae.requires_grad_(False)
    vae.use_spatial_tiling = False
    vae.use_temporal_tiling = False
    vae.use_slicing = False
    return vae


def test_stages_compose_to_the_decoder():
    """Running the stages back to back is exactly ``vae.decode``, frame selection included."""
    vae = _small_vae()
    latents = torch.randn(1, 32, 1, 4, 4)
    with torch.no_grad():
        expected = vae.decode(latents, return_dict=False)[0]
        hidden = latents
        stages = decoder_stages(vae)
        for _, stage in stages:
            hidden = stage(hidden)
    # decode() keeps the last frame for a single-frame latent; StagedDeviceDecoder too.
    actual = hidden[:, :, -1:]
    assert [name for name, _ in stages] == ["in+mid", "up0", "up1", "up2", "up3", "up4", "out"]
    assert actual.shape == expected.shape, f"{tuple(actual.shape)} vs {tuple(expected.shape)}"
    error = (actual - expected).abs().max().item()
    assert error == 0.0, f"staged decode differs by {error:.3e}"
    print(f"7 stages compose to decode() exactly, {tuple(actual.shape)}")


def test_fp32_reductions_match_the_reference_in_float32():
    """In float32 the patched GroupNorm and attention reproduce the stock modules."""
    reference = _small_vae()
    patched = _small_vae()
    norms = install_fp32_norms(patched)
    attention = install_fp32_attention(patched)
    assert norms == sum(isinstance(m, nn.GroupNorm) for m in patched.decoder.modules())
    assert attention == 1

    latents = torch.randn(1, 32, 1, 4, 4)
    with torch.no_grad():
        expected = reference.decode(latents, return_dict=False)[0]
        actual = patched.decode(latents, return_dict=False)[0]
    error = (actual - expected).abs().max().item() / expected.abs().max().item()
    assert error < 1e-4, f"float32 reductions differ by {error:.3e}"
    print(f"{norms} GroupNorms + {attention} attention, float32 relative error {error:.3e}")


def test_float16_norm_survives_activations_past_256():
    """float16 GroupNorm stays finite and accurate where squaring in float16 would not.

    The decoder's last levels carry activations near 9.7e3; anything past 256 squares past
    float16's 65504. Eager CPU float16 forms the square in float16, so it shows the
    overflow a compiler is free to produce: the naive variance goes to inf here.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 64, 2, 16, 16)
    x = (x / x.abs().max() * 9700.0).half()

    naive = x.reshape(1, 32, -1)
    naive = (naive - naive.float().mean(-1, keepdim=True).half()).pow(2)
    assert not bool(torch.isfinite(naive).all()), "expected the naive square to overflow"

    norm = nn.GroupNorm(32, 64, eps=1e-6).half()
    install_fp32_norms(type("Holder", (), {"decoder": norm})())
    with torch.no_grad():
        actual = norm(x).float()
    expected = nn.functional.group_norm(x.float(), 32, eps=1e-6)
    assert bool(torch.isfinite(actual).all()), "float16 GroupNorm overflowed"
    error = (actual - expected).abs().max().item() / expected.abs().max().item()
    assert error < 2e-3, f"float16 GroupNorm differs by {error:.3e}"
    print(f"float16 GroupNorm at peak 9700: finite, relative error {error:.3e}")

    # A constant group has zero variance; its output must be the bias, not inf * 0.
    with torch.no_grad():
        flat = norm(torch.full((1, 64, 2, 4, 4), 3000.0, dtype=torch.float16)).float()
    assert bool(torch.isfinite(flat).all()), "constant group produced inf/nan"


def test_supports_follows_the_kernel_spatial_range():
    """1024x1024 decodes on device; anything past 1024 a side goes back to the host."""
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import StagedDeviceDecoder

    decoder = StagedDeviceDecoder.__new__(StagedDeviceDecoder)
    decoder.ffactor_spatial = 16
    decoder.max_spatial = 1024
    assert decoder.supports(torch.empty(1, 32, 1, 64, 64))
    assert decoder.supports(torch.empty(1, 32, 1, 48, 64))
    assert not decoder.supports(torch.empty(1, 32, 1, 48, 80))


if __name__ == "__main__":
    test_stages_compose_to_the_decoder()
    test_fp32_reductions_match_the_reference_in_float32()
    test_float16_norm_survives_activations_past_256()
    test_supports_follows_the_kernel_spatial_range()
    print("OK")
