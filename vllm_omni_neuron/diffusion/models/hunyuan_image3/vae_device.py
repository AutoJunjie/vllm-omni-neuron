# SPDX-License-Identifier: Apache-2.0
"""Decode HunyuanImage-3.0 latents on a NeuronCore, one compiled graph per decoder stage.

The host float32 decode is ~26 s of a ~72 s warm request. On one core it is ~2.6 s, but
only under four conditions, each found by the previous answer failing
(``examples/hunyuan_image3/check_vae_decode.py`` reproduces every one of them):

* **Split by depth, not by area.** The whole decoder is 10,746,838 instructions at
  1024x1024 against a 10M ceiling, and smaller images or spatial tiles are *more*
  instructions, not fewer. The decoder is a sequential chain, so each depth level compiles
  as its own graph and the intermediates stay on device between them.
* **3x3x3 convolutions through NKI**, with the 8192- and 4096-wide ``UpsampleDCAE`` banks
  split by output channel (:mod:`.vae_nki_conv`).
* **GroupNorm statistics and the attention softmax in float32.** Plain bfloat16 decodes a
  garbage image (1.0265 relative error); the cause is the 16.8M-element reductions, not
  the mantissa. With float32 reductions the image frame is 1.64e-02 off host float32.
* **bfloat16 residency.** float32 holds 13.1 GiB of a core's 24 GiB and runs out at
  ``up3``; bfloat16 with the duplicate conv weights released is 5.0 GiB.

The NEFF cache key is the lowered MLIR plus :data:`COMPILER_ARGS`, and the MLIR carries
each op's source location -- the absolute path and line of every traced frame. So the
check script and the pipeline share this module rather than copies of it, and a cached
NEFF is only reused for this exact file at this exact path: editing a line above a stage,
or importing the plugin from another checkout, is a cold compile (~2 hours at 1024x1024
for the seven stages).
"""

import logging
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=20000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]


def install_fp32_norms(vae) -> int:
    """Compute every decoder GroupNorm's statistics in float32, output in the input dtype.

    This is the half of ``torch.autocast`` that matters here, and the reason bfloat16
    produced a garbage image: the failure is the *reduction*, not the mantissa. A single
    convolution is only 3.2e-03 off in bfloat16, which 44 layers cannot compound into a
    98% error; a GroupNorm can, twice over. Its last level reduces 4 channels x 4 frames
    x 1024^2 ~ 16.8M elements, and a low-precision running sum stops absorbing new terms
    once it exceeds them by ~256x, so the mean and variance come out badly wrong. And
    computing the variance as ``E[x^2] - E[x]^2`` cancels catastrophically when the mean
    dominates, which can drive it to zero or negative before ``rsqrt``.

    So: accumulate in float32 via ``mean(dtype=torch.float32)``, which upcasts during the
    reduction without materialising a float32 copy of ``x`` -- ``x.float()`` would add a
    2.1 GiB temporary at the largest level, which is what put float32 out of memory in
    the first place. Two passes rather than ``E[x^2] - E[x]^2``, with the centring done
    in the input dtype: that rounds each term relative to its own magnitude, so the
    residual bias on the variance is ~(7.8e-03)^2/3, negligible, where cancellation is
    not. ``eps`` is added in float32, which also keeps it off float16's subnormal range
    (1e-6 is below float16's smallest normal, 6.1e-5).
    """

    def forward(self, x):
        shape = x.shape
        batch, channels = shape[0], shape[1]
        grouped = x.reshape(batch, self.num_groups, channels // self.num_groups, *shape[2:])
        dims = tuple(range(2, grouped.dim()))

        mean = grouped.mean(dim=dims, keepdim=True, dtype=torch.float32)
        centred = grouped - mean.to(x.dtype)
        variance = centred.pow(2).mean(dim=dims, keepdim=True, dtype=torch.float32)
        scale = torch.rsqrt(variance + self.eps).to(x.dtype)

        normalised = (centred * scale).reshape(shape)
        affine_shape = (1, channels) + (1,) * (len(shape) - 2)
        return normalised * self.weight.reshape(affine_shape).to(x.dtype) + self.bias.reshape(
            affine_shape
        ).to(x.dtype)

    installed = 0
    for module in vae.decoder.modules():
        if isinstance(module, nn.GroupNorm):
            module.forward = forward.__get__(module, type(module))
            installed += 1
    return installed


def install_fp32_attention(vae) -> int:
    """Run the mid block's attention softmax in float32, leaving its 1x1 convs alone.

    The softmax is another low-precision reduction whose accumulation dtype this side
    cannot confirm, so it gets the same treatment as the GroupNorm statistics. Only the
    reduction is upcast: casting the module's parameters instead fails outright, because
    Lite device tensors cannot be restrided or retyped in place (``Expected self.dtype()
    == dst.dtype()``), and it would not be the part that matters anyway. The block runs
    at 64x64, inside a stage that is 0.24 s of a 29 s decode.
    """
    from einops import rearrange

    def attention(self, hidden):
        hidden = self.norm(hidden)
        query, key, value = self.q(hidden), self.k(hidden), self.v(hidden)
        batch, channels, frames, height, width = query.shape
        pattern = "b c f h w -> b 1 (f h w) c"
        query = rearrange(query, pattern).contiguous().float()
        key = rearrange(key, pattern).contiguous().float()
        value = rearrange(value, pattern).contiguous().float()
        attended = nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.to(hidden.dtype)
        return rearrange(
            attended,
            "b 1 (f h w) c -> b c f h w",
            f=frames,
            h=height,
            w=width,
            c=channels,
            b=batch,
        )

    installed = 0
    for module in vae.decoder.modules():
        if type(module).__name__ == "AttnBlock":
            module.attention = attention.__get__(module, type(module))
            installed += 1
    return installed


def decoder_stages(vae):
    """Split ``vae.decoder.forward`` into sequential stages, in execution order.

    Boundaries follow the chain the decoder already is: ``conv_in`` plus its
    repeat_interleave skip and the three middle blocks, then one stage per upsample
    level, then the output head. Each stage is a plain callable over one tensor, so a
    stage can be compiled, timed and failed independently of the others.
    """
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import swish

    decoder = vae.decoder
    stages = []

    def stage_in(z):
        repeats = decoder.block_out_channels[0] // decoder.z_channels
        h = decoder.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)
        h = decoder.mid.block_1(h)
        h = decoder.mid.attn_1(h)
        return decoder.mid.block_2(h)

    stages.append(("in+mid", stage_in))

    for i_level in range(len(decoder.block_out_channels)):

        def stage_level(h, level=decoder.up[i_level]):
            for block in level.block:
                h = block(h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)
            return h

        stages.append((f"up{i_level}", stage_level))

    def stage_out(h):
        return decoder.conv_out(swish(decoder.norm_out(h)))

    stages.append(("out", stage_out))
    return stages


def compile_stage(name, fn):
    """``torch.compile`` one decoder stage the way every caller must, to share NEFFs."""
    from vllm_neuron.envs import get_compile_backend_name

    return torch.compile(
        fn,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": f"hunyuan_image3_vae_decode_{name}",
            "compiler_args": COMPILER_ARGS,
        },
    )


def prepare_device_decoder(vae, *, min_d_out: int = 1, free_original: bool = True) -> dict:
    """Install the float32 reductions and the NKI convs on a VAE already on device.

    The NKI install has to come after the move: the packed filters land on the device of
    the weights they were packed from. Returns the NKI dispatch summary.
    """
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_nki_conv import (
        install_nki_conv_dispatch,
    )

    install_fp32_norms(vae)
    install_fp32_attention(vae)
    summary = install_nki_conv_dispatch(vae, min_d_out=min_d_out, free_original=free_original)
    vae.requires_grad_(False)
    return summary


class StagedDeviceDecoder:
    """The VAE decoder on one NeuronCore: bfloat16, float32 reductions, staged graphs.

    Built from the pipeline's host VAE, so the checkpoint is read once. Decoder only: the
    encoder is 1.45 GiB that a decode never touches. Each stage compiles on its first call
    for a given latent shape, so the first request at a new resolution pays the compile
    (~95 s with a warm NEFF cache at 1024x1024, ~2 hours cold).
    """

    def __init__(self, host_vae, vae_config, device, dtype=torch.bfloat16):
        from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

        from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_nki_conv import _MAX_SPATIAL

        start = time.perf_counter()
        self.device = device
        self.dtype = dtype
        self.ffactor_spatial = int(vae_config["ffactor_spatial"])
        self.max_spatial = _MAX_SPATIAL

        vae = AutoencoderKLConv3D.from_config({**vae_config, "only_decoder": True})
        vae = vae.to(dtype).eval()
        # Inference only. Without this the compiled decode traces a backward pass and the
        # Lite backend rejects it with "neuron backend doesn't support events".
        vae.requires_grad_(False)
        vae.use_spatial_tiling = False
        vae.use_temporal_tiling = False
        vae.use_slicing = False
        state = {
            name: tensor.detach().to(dtype)
            for name, tensor in host_vae.state_dict().items()
            if name.startswith("decoder.")
        }
        missing = sorted(set(vae.state_dict()) - set(state))
        if missing:
            raise RuntimeError(f"Device VAE weights missing from the host VAE: {missing[:6]}")
        vae.load_state_dict(state, strict=False, assign=True)
        del state
        self.vae = vae.to(device)

        summary = prepare_device_decoder(self.vae)
        # The five upsample levels share one code object (``stage_level``), and Dynamo
        # counts every level at every resolution against that object's recompile limit
        # (8 by default) -- a second resolution would exceed it and run eagerly.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        self.stages = [(name, compile_stage(name, fn)) for name, fn in decoder_stages(self.vae)]
        logger.info(
            "HunyuanImage3: VAE decoder on %s in %s, %d NKI convs (%d on compiler, %.2f GiB "
            "of duplicate weights released), %d stages, built in %.1fs",
            device,
            dtype,
            summary["routed"],
            summary["skipped"],
            summary["freed_bytes"] / 2**30,
            len(self.stages),
            time.perf_counter() - start,
        )

    def supports(self, latents: torch.Tensor) -> bool:
        """Whether this latent shape decodes inside the NKI conv's spatial range.

        The last level convolves at full image resolution and the kernel covers up to
        1024 per side. Past that a routed site would need its torch fallback, whose
        weight was released, so the caller decodes such shapes on the host instead.
        """
        height, width = latents.shape[-2:]
        return max(height, width) * self.ffactor_spatial <= self.max_spatial

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``vae.decode(latents)[0]`` for ``(B, C, T, h, w)`` latents, returned float32 on host."""
        # Cast on the host and transfer separately: Lite copies require matching dtypes.
        hidden = latents.to(self.dtype).to(self.device)
        with torch.no_grad():
            for _, stage in self.stages:
                hidden = stage(hidden)
        decoded = hidden.to("cpu").float()
        # Same as AutoencoderKLConv3D.decode: a single-frame latent keeps the last frame.
        if latents.shape[-3] == 1:
            decoded = decoded[:, :, -1:]
        return decoded
