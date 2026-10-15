# SPDX-License-Identifier: Apache-2.0
"""Decode HunyuanImage-3.0 latents on a NeuronCore, one compiled graph per decoder stage.

The host float32 decode is ~26 s of a ~72 s warm request. On one core it is ~2.6 s, but
only under four conditions, each found by the previous answer failing
(each one reproduced on device during bring-up):

* **Split by depth, not by area.** The whole decoder is 10,746,838 instructions at
  1024x1024 against a 10M ceiling, and smaller images or spatial tiles are *more*
  instructions, not fewer. The decoder is a sequential chain, so each depth level compiles
  as its own graph and the intermediates stay on device between them.
* **3x3x3 convolutions through NKI**, with the 8192- and 4096-wide ``UpsampleDCAE`` banks
  split by output channel (:mod:`.vae_nki_conv`).
* **GroupNorm statistics and the attention softmax in float32.** Plain bfloat16 decodes a
  garbage image (1.0265 relative error); the cause is the 16.8M-element reductions, not
  the mantissa. With float32 reductions the image frame is 1.64e-02 off host float32.
* **16-bit residency.** float32 holds 13.1 GiB of a core's 24 GiB and runs out at
  ``up3``; bfloat16 with the duplicate conv weights released is 5.0 GiB.

The default dtype is float16: same speed and residency as bfloat16, with 3 more mantissa
bits -- 66.15 dB PSNR to the host float32 decode against bfloat16's 58.71 dB. Its range
is the price. Activations peak at ~9.7e3 here, inside float16's 65504, but a GroupNorm
squares them; see :func:`install_fp32_norms`.

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

    In float16 the square itself is the hazard: anything past 256 squares past 65504, and
    the last two levels carry activations near 9.7e3. Accumulating in float32 does not
    help when the square is formed first, in the input dtype. So float16 scales each
    group by a power of two ``s`` that brings ``|x - mean|`` to at most 1 *before*
    centring, and normalises the scaled values directly -- the result is scale-invariant
    once ``eps`` is scaled by ``s**2`` too. A power of two is exact in every binary format,
    so this adds no rounding, nothing larger than 1 is ever squared, and there is still
    exactly one full-size temporary. That last part is a measured constraint: scaling the
    centred values instead kept a second one alive, which grew the largest stage's
    scratchpad by 0.75 GiB and ran the output rank's core out of memory once all seven
    stage NEFFs were resident (the second request). The peak comes from ``amax``/``amin``
    rather than ``abs().amax()`` for the same reason. The rsqrt is clamped to float16's
    range only for a constant group, where every centred value is 0 anyway.
    """

    def forward(self, x):
        shape = x.shape
        batch, channels = shape[0], shape[1]
        grouped = x.reshape(batch, self.num_groups, channels // self.num_groups, *shape[2:])
        dims = tuple(range(2, grouped.dim()))

        mean = grouped.mean(dim=dims, keepdim=True, dtype=torch.float32)
        if x.dtype == torch.float16:
            peak = torch.maximum(
                grouped.amax(dim=dims, keepdim=True), -grouped.amin(dim=dims, keepdim=True)
            ).float()
            unit = torch.exp2(-torch.ceil(torch.log2(2.0 * peak.clamp_min(2.0**-14))))
            centred = grouped * unit.to(x.dtype) - (mean * unit).to(x.dtype)
            variance = centred.pow(2).mean(dim=dims, keepdim=True, dtype=torch.float32)
            scale = torch.rsqrt(variance + self.eps * unit * unit).clamp(max=65504.0)
            scale = scale.to(x.dtype)
        else:
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
    """The VAE decoder on one NeuronCore: 16-bit, float32 reductions, staged graphs.

    Built from the pipeline's host VAE, so the checkpoint is read once. Decoder only: the
    encoder is 1.45 GiB that a decode never touches. Each stage compiles on its first call
    for a given latent shape, so the first request at a new resolution pays the compile
    (~95 s with a warm NEFF cache at 1024x1024, ~2 hours cold).
    """

    def __init__(self, host_vae, vae_config, device, dtype=torch.float16):
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


# --- the decode split across ranks by rows ---------------------------------------------
#
# Below the single-core decoder on purpose: its stage NEFFs are keyed on the lines above.


def shard_plan(vae, latent_height: int, parts: int) -> list[dict]:
    """Each stage's input band of rows for an ``parts``-way split of the image height.

    Rank ``r`` owns rows ``[r * c, (r + 1) * c)`` of every resolution (``c`` = that
    resolution's height / ``parts``) and also computes a halo past them, wide enough that
    its owned rows come out exact: every convolution here is 3x3 (radius 1) with zero
    padding, so a band's rows go stale one row per convolution from a cut edge inwards.
    Walking back from the output, a stage's input halo is what the next stage needs,
    mapped through this stage's upsample (``ceil(h / 2)``), plus one row per convolution
    the stage runs at its input resolution. ``in+mid`` is not split: its attention spans
    the whole 64x64 latent, and it is 0.01 s of the decode.

    Returns, per stage in execution order, ``{"name", "height", "core", "halo"}`` with
    the stage's input height, owned rows and halo at the stage's input resolution.
    """
    decoder = vae.decoder
    levels = len(decoder.block_out_channels)
    heights, radii, factors = [], [], []
    height = latent_height
    for i_level in range(levels):
        level = decoder.up[i_level]
        convs = 2 * len(level.block)
        upsample = hasattr(level, "upsample")
        heights.append(height)
        radii.append(convs + int(upsample))
        factors.append(2 if upsample else 1)
        height *= 2 if upsample else 1
    halo_out = 1  # conv_out
    plan = [{"name": "out", "height": height, "halo": halo_out}]
    halo = halo_out
    for i_level in reversed(range(levels)):
        halo = -(-halo // factors[i_level]) + radii[i_level]
        plan.append({"name": f"up{i_level}", "height": heights[i_level], "halo": halo})
    plan.append({"name": "in+mid", "height": latent_height, "halo": None})
    plan.reverse()
    for stage in plan:
        if stage["height"] % parts:
            raise ValueError(f"{stage['name']} height {stage['height']} does not split {parts} ways")
        stage["core"] = stage["height"] // parts
    return plan


def band(stage: dict, rank: int, parts: int) -> tuple[int, int]:
    """Rows ``[start, stop)`` of the stage's input that rank ``rank`` computes."""
    if stage["halo"] is None:
        return 0, stage["height"]
    core, halo = stage["core"], stage["halo"]
    return max(0, rank * core - halo), min(stage["height"], (rank + 1) * core + halo)


def install_sharded_norms(vae, plan: list[dict], rank: int, parts: int, group, pairs=None) -> int:
    """GroupNorms below the middle block with statistics over every rank's owned rows.

    Each rank sums only the rows it owns, so the all-reduced sums cover the image exactly
    once and every rank normalises its whole band -- halo included -- with the statistics
    of the full image. Same two-pass float32 reduction as :func:`install_fp32_norms`; in
    float16 a rank scales by its own power of two (it bounds this rank's values and the
    already-global mean, so nothing past 1 is squared), and its squared sum is divided
    back to the unscaled value before the all-reduce, which is exact for a power of two.
    Two all-reduces of ``[B, groups]`` float32 per GroupNorm.
    """
    import torch.distributed as dist

    stages = {stage["name"]: stage for stage in plan}

    def forward(self, x):
        shape = x.shape
        batch, channels = shape[0], shape[1]
        grouped = x.reshape(batch, self.num_groups, channels // self.num_groups, *shape[2:])
        dims = tuple(range(2, grouped.dim()))
        lo, hi = self._owned_rows
        owned = grouped[..., lo:hi, :]
        count = owned[0, 0].numel() * self._parts

        total = owned.sum(dim=dims, keepdim=True, dtype=torch.float32)
        dist.all_reduce(total, group=self._group)
        mean = total / count
        if x.dtype == torch.float16:
            peak = torch.maximum(
                torch.maximum(
                    grouped.amax(dim=dims, keepdim=True), -grouped.amin(dim=dims, keepdim=True)
                ).float(),
                mean.abs(),
            )
            unit = torch.exp2(-torch.ceil(torch.log2(2.0 * peak.clamp_min(2.0**-14))))
            centred = grouped * unit.to(x.dtype) - (mean * unit).to(x.dtype)
        else:
            unit = None
            centred = grouped - mean.to(x.dtype)
        squares = centred[..., lo:hi, :].pow(2).sum(dim=dims, keepdim=True, dtype=torch.float32)
        if unit is not None:
            squares = squares / (unit * unit)
        dist.all_reduce(squares, group=self._group)
        variance = squares / count
        if unit is not None:
            scale = torch.rsqrt((variance + self.eps) * unit * unit).clamp(max=65504.0)
        else:
            scale = torch.rsqrt(variance + self.eps)
        normalised = (centred * scale.to(x.dtype)).reshape(shape)
        affine_shape = (1, channels) + (1,) * (len(shape) - 2)
        return normalised * self.weight.reshape(affine_shape).to(x.dtype) + self.bias.reshape(
            affine_shape
        ).to(x.dtype)

    def owned_rows(stage):
        start, _ = band(stage, rank, parts)
        lo = rank * stage["core"] - start
        return lo, lo + stage["core"]

    decoder = vae.decoder  # ``pairs``: (module, stage name) to shard, default the decoder's
    pairs = pairs or [(decoder.up[i], f"up{i}") for i in range(len(decoder.block_out_channels))] + [(decoder.norm_out, "out")]
    targets = [(module, stages[name]) for module, name in pairs]
    installed = 0
    for parent, stage in targets:
        for module in parent.modules():
            if isinstance(module, nn.GroupNorm):
                module._owned_rows = owned_rows(stage)
                module._parts = parts
                module._group = group
                module.forward = forward.__get__(module, type(module))
                installed += 1
    return installed


def sharded_stages(vae, plan: list[dict], rank: int, parts: int, group):
    """:func:`decoder_stages` for this rank's band: crop to the band, run, gather at the end.

    Rank 0's band starts at the image top and the last rank's ends at the bottom, so each
    stage's input is a slice of the previous stage's output at known offsets. The output
    stage keeps the owned rows and all-gathers them, so every rank returns the image.
    """
    import torch.distributed as dist

    full = dict(decoder_stages(vae))
    stages = []
    previous_start, previous_factor = 0, 1
    for index, stage in enumerate(plan):
        name, fn = stage["name"], full[stage["name"]]
        start, stop = band(stage, rank, parts)
        offset = start - previous_start * previous_factor
        rows = stop - start
        last = index == len(plan) - 1

        def run(h, fn=fn, offset=offset, rows=rows, last=last, core=stage["core"], start=start):
            h = fn(h[..., offset : offset + rows, :])
            if not last:
                return h
            own = h[..., rank * core - start : rank * core - start + core, :].contiguous()
            out = own.new_empty((parts * own.shape[0], *own.shape[1:]))
            dist.all_gather_into_tensor(out, own, group=group)
            return torch.cat(out.reshape(parts, *own.shape).unbind(0), dim=-2)

        stages.append((name, run))
        previous_start = start
        if index + 1 < len(plan):
            previous_factor = plan[index + 1]["height"] // stage["height"]
    return stages


class ShardedDeviceDecoder(StagedDeviceDecoder):
    """:class:`StagedDeviceDecoder` split ``parts`` ways by image rows across ranks.

    Every rank in ``group`` calls :meth:`decode` with the same latents; each computes its
    band of rows (:func:`shard_plan`) on its own core and all of them return the whole
    image. With two bands on the backbone's own cores, the VAE's ~14 GiB on one core
    becomes ~half on each, which fits beside a TP16 backbone shard.
    """

    def __init__(self, host_vae, vae_config, device, dtype, rank: int, parts: int, group):
        super().__init__(host_vae, vae_config, device, dtype=dtype)
        self.rank, self.parts, self.group = rank, parts, group
        self._plans: dict = {}

    def supports(self, latents: torch.Tensor) -> bool:
        """The single-core range, with a latent height that splits into equal bands."""
        return super().supports(latents) and latents.shape[-2] % self.parts == 0

    def _stages_for(self, latent_height: int):
        if latent_height not in self._plans:
            plan = shard_plan(self.vae, latent_height, self.parts)
            install_sharded_norms(self.vae, plan, self.rank, self.parts, self.group)
            self._plans[latent_height] = [
                (name, compile_stage(f"{name}_rows{self.rank}of{self.parts}", fn))
                for name, fn in sharded_stages(self.vae, plan, self.rank, self.parts, self.group)
            ]
        return self._plans[latent_height]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = latents.to(self.dtype).to(self.device)
        with torch.no_grad():
            for _, stage in self._stages_for(int(latents.shape[-2])):
                hidden = stage(hidden)
        decoded = hidden.to("cpu").float()
        if latents.shape[-3] == 1:
            decoded = decoded[:, :, -1:]
        return decoded


def vae_row_groups(parts: int):
    """The groups of the ``vae_parallel`` row split: ``parts``-rank blocks over the world.

    Every rank calls it (``new_group`` is collective), and per block it creates a device
    group and its gloo twin, in that order. The blocks are registered as the device
    groups' replica groups, which the split decode's graphs carry -- so a precompile
    (``compile_vae_stages.py --vae-parallel``) builds the same ones and hits its NEFFs.
    Returns this rank's ``(device group, host group)``.
    """
    import torch.distributed as dist

    from vllm_omni_neuron.lite_compat import register_process_group_replica_groups

    world, rank = dist.get_world_size(), dist.get_rank()
    if world % parts:
        raise ValueError(f"vae_parallel={parts} must divide the {world} ranks")
    blocks = [list(range(base, base + parts)) for base in range(0, world, parts)]
    for block in blocks:
        groups = dist.new_group(block), dist.new_group(block, backend="gloo")
        if rank in block:
            own = groups
    register_process_group_replica_groups(own[0].group_name, blocks)
    return own


def bootstrap_device_collectives(device) -> None:
    """One compiled all-reduce over every rank of the world, called on every rank.

    A rank's first device collective joins the world's communicator bootstrap (through
    global rank 0), which waits for every rank. In the pipeline the backbone's first graph
    does that on all ranks; a process that runs only the row-split decode, on some ranks
    (``compile_vae_stages.py --vae-parallel``, the tier-1 test), calls this first.
    """
    import torch.distributed as dist

    from vllm_omni_neuron.lite_compat import register_process_group_replica_groups

    ranks = list(range(dist.get_world_size()))
    group = dist.new_group(ranks)
    register_process_group_replica_groups(group.group_name, [ranks])

    def reduce(x):
        dist.all_reduce(x, group=group)
        return x

    compile_stage("world_bootstrap", reduce)(torch.ones(1).to(device)).to("cpu")
