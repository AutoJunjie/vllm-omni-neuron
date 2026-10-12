# SPDX-License-Identifier: Apache-2.0
"""The VAE's NKI conv dispatch must be exact, and must route the sites it claims to.

The dispatch replaces the decoder's 3x3x3 convolutions with an ``nkilib`` kernel, and
splits any filter bank wider than the kernel's ``C_out`` range into groups. Both of those
are easy to get subtly wrong -- a transposed pack or a mis-sliced bias produces a
plausible image rather than an error -- so this asserts the packed-and-split form
reproduces ``nn.Conv3d`` exactly, on CPU, without a device or the kernel itself.

    VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_vae_nki_conv.py
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_nki_conv import (
    _MAX_OUT_CHANNELS,
    _is_eligible,
    _pack_filters,
    _split_out_channels,
)


def _replay_packed(x, filters, biases):
    """Run the packed-and-split filters through ``F.conv3d``, unpacking each group.

    The NKI kernel takes ``[K_d, K_h, K_w, C_in, C_out]``; this inverts that permutation
    so the test exercises the real packing rather than a restatement of it.
    """
    outputs = []
    for packed, bias in zip(filters, biases):
        weight = packed.permute(4, 3, 0, 1, 2).contiguous()
        outputs.append(F.conv3d(x, weight, bias, padding=1))
    return torch.cat(outputs, dim=1)


def test_pack_is_exact_without_splitting():
    """A bank inside the kernel's range packs to one group and stays exact."""
    torch.manual_seed(0)
    conv = nn.Conv3d(128, 128, 3, padding=1)
    x = torch.randn(1, 128, 2, 9, 7)

    filters, biases = _pack_filters(conv.weight, conv.bias)
    assert len(filters) == 1, f"expected a single group, got {len(filters)}"
    assert tuple(filters[0].shape) == (3, 3, 3, 128, 128)

    error = (_replay_packed(x, filters, biases) - conv(x)).abs().max().item()
    assert error < 1e-5, f"packed conv differs by {error:.3e}"
    print(f"no-split pack exact to {error:.3e}")


def test_split_is_exact_for_the_upsample_bank():
    """The ``UpsampleDCAE`` shape that broke the compiler must split and stay exact.

    ``up.0.upsample.conv`` is 1024 -> 8192, and on its own it generated 14,155,776
    instructions against a 300,000 per-operator limit (``NCC_EXTP003``). C_in is reduced
    here to keep the test cheap; the ``C_out`` width is what the split keys on.
    """
    torch.manual_seed(0)
    out_channels = 8192
    conv = nn.Conv3d(96, out_channels, 3, padding=1)
    x = torch.randn(1, 96, 1, 4, 4)

    filters, biases = _pack_filters(conv.weight, conv.bias)
    expected_groups = -(-out_channels // _MAX_OUT_CHANNELS)
    assert len(filters) == expected_groups, f"{len(filters)} groups, want {expected_groups}"
    assert sum(f.shape[-1] for f in filters) == out_channels
    assert sum(b.numel() for b in biases) == out_channels
    assert all(f.shape[-1] <= _MAX_OUT_CHANNELS for f in filters)
    assert all(f.is_contiguous() for f in filters), "device cannot restride a view in place"

    error = (_replay_packed(x, filters, biases) - conv(x)).abs().max().item()
    assert error < 1e-4, f"split conv differs by {error:.3e}"
    print(f"{out_channels} -> {[f.shape[-1] for f in filters]}, exact to {error:.3e}")


def test_group_boundaries_follow_the_bias():
    """A bias sliced out of step with the filters would shift whole channel groups."""
    torch.manual_seed(0)
    conv = nn.Conv3d(96, 4096, 3, padding=1)
    filters, biases = _pack_filters(conv.weight, conv.bias)

    offset = 0
    for packed, bias in zip(filters, biases):
        width = packed.shape[-1]
        assert torch.equal(bias, conv.bias.detach()[offset : offset + width])
        assert torch.equal(
            packed, conv.weight.detach().permute(2, 3, 4, 1, 0)[..., offset : offset + width]
        )
        offset += width
    assert offset == 4096
    print(f"{len(filters)} groups aligned with the bias")


def test_eligibility_matches_the_decoder_sites():
    """The gate must route the resblock/upsample convs and leave the rest alone."""
    # (in, out, kernel, padding, expected) -- the real shapes from the decoder.
    cases = [
        (1024, 1024, 3, 1, True),  # mid/up resblock conv1, conv2
        (512, 1024, 3, 1, True),  # up.2.upsample.conv
        (1024, 8192, 3, 1, True),  # up.0.upsample.conv -- splits, must not be rejected
        (32, 1024, 3, 1, False),  # conv_in: C_in below the contraction floor
        (128, 3, 3, 1, False),  # conv_out: C_out starves the PSUM partition dim
        (1024, 1024, 1, 0, False),  # attn q/k/v/proj_out: 1x1, not this kernel's shape
    ]
    for in_channels, out_channels, kernel, padding, expected in cases:
        conv = nn.Conv3d(in_channels, out_channels, kernel, padding=padding)
        actual = _is_eligible(conv)
        assert actual is expected, (
            f"{in_channels}->{out_channels} k{kernel} p{padding}: "
            f"eligible={actual}, expected {expected}"
        )
    print(f"{len(cases)} eligibility cases match")


if __name__ == "__main__":
    test_pack_is_exact_without_splitting()
    test_split_is_exact_for_the_upsample_bank()
    test_group_boundaries_follow_the_bias()
    test_eligibility_matches_the_decoder_sites()
    print("OK")
