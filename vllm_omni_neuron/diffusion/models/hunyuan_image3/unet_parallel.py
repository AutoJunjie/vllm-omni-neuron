# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel patch embed and final layer for the HunyuanImage-3.0 denoise step.

The ``UNetDown`` patch embed and the ``UNetUp`` final layer are ResBlocks over the latent
grid with up to 4096 channels -- ~4 TFLOP per CFG-batched 1024x1024 step -- and every rank
used to run them whole, plus a full [B, HW, C] <-> [B, C, H, W] transpose of the 67 MB
token tensor at the tail. A device profile of a warm step put them at ~245 ms of 760 ms
(66 ms ahead of the first layer, 179 ms after the last one).

Every GroupNorm in them has 32 groups, so splitting channels by group keeps each rank's
statistics local: rank ``r`` of ``world`` owns groups ``[r * g, (r + 1) * g)``,
``g = 32 // world``. Each convolution is column-parallel -- its input channels are
all-gathered, each rank computes its own output channels -- and the timestep projection
produces only the rank's scale/shift slice. Per output channel the arithmetic is the
replicated path's (``_res_block`` in ``pipeline_hunyuan_image3.py``): GroupNorm,
modulation and SiLU in float32, convolutions in the weights' dtype.

The token transposes shrink to the rank's own channel slice, except for the one that
turns the gathered patch-embed output into tokens.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

_GROUPS = 32

# ``gather(t)`` returns every rank's ``t`` stacked on a new leading dim: [world, *t.shape].
Gather = Callable[[torch.Tensor], torch.Tensor]


def _slice(channels: int, rank: int, world: int) -> slice:
    per = channels // world
    return slice(rank * per, (rank + 1) * per)


def _frozen(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor.detach().clone().contiguous(), requires_grad=False)


def _group_norm(x: torch.Tensor, weight, bias, groups: int, eps: float) -> torch.Tensor:
    """GroupNorm of this rank's whole groups, in float32 (see ``_group_norm_f32``)."""
    return F.group_norm(x.float(), groups, weight.float(), bias.float(), eps)


def _channels(gathered: torch.Tensor) -> torch.Tensor:
    """[world, B, c, H, W] -> [B, world * c, H, W], rank-major as the channels are split."""
    world, bsz, channels, height, width = gathered.shape
    return gathered.permute(1, 0, 2, 3, 4).reshape(bsz, world * channels, height, width)


class _ShardedResBlock(nn.Module):
    """One rank's slice of a non-resampling ``ResBlock`` with a 1x1 skip convolution."""

    def __init__(self, block: nn.Module, rank: int, world: int) -> None:
        super().__init__()
        in_norm, in_conv = block.in_layers[0], block.in_layers[2]
        emb_linear = block.emb_layers[1]
        out_norm, out_conv = block.out_layers[0], block.out_layers[3]
        skip = block.skip_connection
        in_channels, out_channels = in_conv.in_channels, in_conv.out_channels
        own_in, own_out = _slice(in_channels, rank, world), _slice(out_channels, rank, world)
        # emb_layers yields [scale | shift] over out_channels; keep this rank's rows of each.
        own_emb = torch.cat(
            (
                torch.arange(out_channels)[own_out],
                torch.arange(out_channels, 2 * out_channels)[own_out],
            )
        )

        self.groups = _GROUPS // world
        self.in_eps, self.out_eps = float(in_norm.eps), float(out_norm.eps)
        self.in_norm_weight, self.in_norm_bias = _frozen(in_norm.weight[own_in]), _frozen(in_norm.bias[own_in])
        self.in_weight, self.in_bias = _frozen(in_conv.weight[own_out]), _frozen(in_conv.bias[own_out])
        self.emb_weight, self.emb_bias = _frozen(emb_linear.weight[own_emb]), _frozen(emb_linear.bias[own_emb])
        self.out_norm_weight = _frozen(out_norm.weight[own_out])
        self.out_norm_bias = _frozen(out_norm.bias[own_out])
        self.out_weight, self.out_bias = _frozen(out_conv.weight[own_out]), _frozen(out_conv.bias[own_out])
        # The 1x1 skip as a [own_out, in_channels] matrix, applicable to either layout.
        self.skip_weight = _frozen(skip.weight[own_out, :, 0, 0])
        self.skip_bias = _frozen(skip.bias[own_out])

    def in_layers(self, x: torch.Tensor) -> torch.Tensor:
        """GroupNorm + SiLU of this rank's input channels, cast for the convolution."""
        normed = _group_norm(x, self.in_norm_weight, self.in_norm_bias, self.groups, self.in_eps)
        return F.silu(normed).to(self.in_weight.dtype)

    def out_layers(self, h_full: torch.Tensor, emb: torch.Tensor, gather: Gather) -> torch.Tensor:
        """in conv -> adaptive GroupNorm -> SiLU -> out conv, for this rank's channels."""
        h = F.conv2d(h_full, self.in_weight, self.in_bias, padding=1)
        emb_out = F.linear(F.silu(emb), self.emb_weight, self.emb_bias)
        scale, shift = torch.chunk(emb_out.float()[..., None, None], 2, dim=1)
        normed = _group_norm(h, self.out_norm_weight, self.out_norm_bias, self.groups, self.out_eps)
        h = F.silu(normed * (1.0 + scale) + shift).to(self.out_weight.dtype)
        return F.conv2d(_channels(gather(h)), self.out_weight, self.out_bias, padding=1)


class UNetShard(nn.Module):
    """This rank's slices of the patch embed (``UNetDown``) and final layer (``UNetUp``)."""

    def __init__(self, patch_embed: nn.Module, final_layer: nn.Module, rank: int, world: int) -> None:
        super().__init__()
        check_supported(patch_embed, final_layer, world)
        self.rank, self.world = rank, world
        self.groups = _GROUPS // world
        conv0, down = patch_embed.model[0], patch_embed.model[1]
        up, head = final_layer.model[0], final_layer.model[1]
        own0 = _slice(conv0.out_channels, rank, world)
        self.conv0_weight, self.conv0_bias = _frozen(conv0.weight[own0]), _frozen(conv0.bias[own0])
        self.down = _ShardedResBlock(down, rank, world)
        self.up = _ShardedResBlock(up, rank, world)
        head_norm, head_conv = head[0], head[2]
        own_head = _slice(head_norm.num_channels, rank, world)
        self.head_eps = float(head_norm.eps)
        self.head_norm_weight = _frozen(head_norm.weight[own_head])
        self.head_norm_bias = _frozen(head_norm.bias[own_head])
        # The output convolution is small (1024 -> 32 channels) and stays whole.
        self.head_weight, self.head_bias = _frozen(head_conv.weight), _frozen(head_conv.bias)
        self.group = None

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Stack every TP rank's ``tensor`` on a new leading dim, inside the compiled graph."""
        out = tensor.new_empty((self.world * tensor.shape[0], *tensor.shape[1:]))
        dist.all_gather_into_tensor(out, tensor.contiguous(), group=self.group)
        return out.reshape(self.world, *tensor.shape)


def check_supported(patch_embed: nn.Module, final_layer: nn.Module, world: int) -> None:
    """Raise unless the module layout is the patch_size=1 one this split is written for."""
    if world < 1 or _GROUPS % world:
        raise ValueError(f"tensor_parallel_size={world} must divide the {_GROUPS} GroupNorm groups")
    if len(patch_embed.model) != 2 or len(final_layer.model) != 2:
        raise ValueError("expected patch_size=1: one conv + one ResBlock / one ResBlock + out_norm")
    blocks = (patch_embed.model[1], final_layer.model[0])
    for block in blocks:
        if getattr(block, "updown", False) or not isinstance(block.skip_connection, nn.Conv2d):
            raise ValueError("expected non-resampling ResBlocks with a 1x1 skip convolution")
    channels = [patch_embed.model[0].out_channels]
    for block in blocks:
        channels += [block.in_layers[2].in_channels, block.in_layers[2].out_channels]
    if any(c % _GROUPS for c in channels):
        raise ValueError(f"channel counts {channels} must be multiples of {_GROUPS}")


def unet_down_tokens(shard: UNetShard, latents: torch.Tensor, t_emb: torch.Tensor, gather: Gather):
    """``_unet_down_tokens`` with the work split by channel group across the TP ranks."""
    bsz, _, height, width = latents.shape
    x = F.conv2d(latents, shard.conv0_weight, shard.conv0_bias, padding=1)
    h = shard.down.in_layers(x)
    # The skip needs the conv0 output and the in conv its normalised form: one gather.
    both = gather(torch.cat((x.to(h.dtype), h), dim=1))
    world, _, channels2 = both.shape[:3]
    both = both.reshape(world, bsz, 2, channels2 // 2, height, width)
    x_full, h_full = _channels(both[:, :, 0]), _channels(both[:, :, 1])
    h = shard.down.out_layers(h_full, t_emb, gather)
    skip = F.conv2d(x_full, shard.down.skip_weight[..., None, None], shard.down.skip_bias)
    y_full = _channels(gather(skip + h))
    channels = y_full.shape[1]
    return y_full.reshape(bsz, channels, height * width).transpose(1, 2), height, width


def unet_up_image(
    shard: UNetShard,
    tokens: torch.Tensor,
    t_emb: torch.Tensor,
    token_h: int,
    token_w: int,
    gather: Gather,
) -> torch.Tensor:
    """``_unet_up_image`` with the work split by channel group across the TP ranks.

    ``tokens`` is whole on every rank (the backbone's all-reduced output), so the rank
    reads its own input channels straight from it and only that slice is transposed.
    """
    bsz, _, channels = tokens.shape
    own = _slice(channels, shard.rank, shard.world)
    x = tokens[:, :, own].transpose(1, 2).reshape(bsz, own.stop - own.start, token_h, token_w)
    h_full = _channels(gather(shard.up.in_layers(x)))
    h = shard.up.out_layers(h_full, t_emb, gather)
    skip = F.linear(tokens, shard.up.skip_weight, shard.up.skip_bias)
    skip = skip.transpose(1, 2).reshape(bsz, -1, token_h, token_w)
    y = skip + h
    z = F.silu(
        _group_norm(y, shard.head_norm_weight, shard.head_norm_bias, shard.groups, shard.head_eps)
    ).to(shard.head_weight.dtype)
    return F.conv2d(_channels(gather(z)), shard.head_weight, shard.head_bias, padding=1)
