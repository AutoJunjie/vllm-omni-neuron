# SPDX-License-Identifier: Apache-2.0
"""Can the HunyuanImage-3.0 VAE decoder run on device, and is it faster there?

The decode is the largest remaining term in a warm request — ~26 s of float32 on the
host, 36% of end to end — because this pipeline keeps the VAE off the device. This script
answers both halves of the question before the pipeline is restructured around it: it
loads the real VAE weights, decodes the model's actual latent shape on the host as a
reference, then compiles the same decode for Neuron and reports the error and the speedup.

``--stages`` is the mode that matters. The whole decoder graph is over the compiler's
10M instruction ceiling, and shrinking the *image* makes that worse, not better --
10,746,838 instructions at 1024x1024, 11,852,476 at 512x512, and 17,488,800 for a single
384x384 spatial tile -- because the count tracks the decoder's op count and per-op
overhead at small spatial dims, not the data volume. (Spatial tiling is wrong on the host
too: 56.98 s tiled vs 29.13 s whole, from decoding 16 overlapping tiles.) So split by
depth instead: the decoder is a sequential chain, and each stage compiles as its own
graph with a fraction of the ops at full spatial size. A stage that fails to compile is
reported and skipped rather than aborting the run, so one pass maps every stage.

Two things in this VAE are known risks for a fullgraph trace, and the output says which
one bites:

* its custom ``Conv3d`` splits the temporal axis when an input exceeds 2 GB, and that
  path mutates ``self.padding`` and writes into slices of a freshly padded tensor — the
  same in-graph-mutation shape that made ``neuronx-cc``'s MaskPropagation pass fail for
  the MoE. At 1024x1024 the largest activation should stay under the threshold, so the
  split should never trigger; ``--report-conv-splits`` checks that rather than assuming.
* ``AttnBlock`` and ``UpsampleDCAE`` use einops ``rearrange``.

    python examples/hunyuan_image3/check_vae_decode.py --stages
    python examples/hunyuan_image3/check_vae_decode.py --dtype bfloat16 --height 512
"""

import argparse
import gc
import json
import os
import re
import time

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
# The decoder graph is ~10.7M instructions, over the compiler's default 10M ceiling
# (NCC_EVRF007). NEURON_CC_FLAGS is the channel the Lite backend actually passes through,
# and setdefault is useless here because the launcher already exports it — so append the
# raised limit to whatever is set. Mirrors what the Wan2.2 VAE does.
_cc_flags = os.environ.get("NEURON_CC_FLAGS", "-O1 --hbm-scratchpad-page-size=2048")
if "--internal-max-instruction-limit" not in _cc_flags:
    _cc_flags += " --internal-max-instruction-limit=20000000"
os.environ["NEURON_CC_FLAGS"] = _cc_flags
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "600")

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch  # noqa: E402

torch.nn.functional.gelu = torch.ops.aten.gelu.default

from vllm.transformers_utils.config import get_config  # noqa: E402
from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
    initialize as initialize_lite,
)

_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=20000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]

parser = argparse.ArgumentParser(description="HunyuanImage-3.0 VAE decode on device")
parser.add_argument("--model-path", default="/workspace/models/HunyuanImage-3.0-Instruct")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument(
    "--dtype",
    default="float32",
    choices=["float32", "bfloat16"],
    help="Device dtype. The host reference is always float32.",
)
parser.add_argument("--runs", type=int, default=3, help="Timed runs on each side")
parser.add_argument("--tolerance", type=float, default=2e-2)
parser.add_argument(
    "--report-conv-splits",
    action="store_true",
    help="Count how many Conv3d calls take the >2 GB temporal-split path (which mutates "
    "module state and writes into slices — the pattern to avoid in a traced graph).",
)
parser.add_argument(
    "--tile",
    action="store_true",
    help="Decode in spatial tiles. The whole-image graph is 10.7M instructions at\n"
    "1024x1024 and 11.9M at 512x512 — over the compiler's 10M threshold either way,\n"
    "with NCC_IXTP002 suggesting tiling. Tiling also forces fullgraph=False, since\n"
    "boundary tiles have their own shapes and so recompile.",
)
parser.add_argument(
    "--stages",
    action="store_true",
    help="Compile the decoder as one graph per stage (conv_in+mid, each upsample level, "
    "norm_out+conv_out) instead of one graph for the whole decode. Intermediates stay on "
    "device between stages. Reports per-stage host time and compiled instruction count.",
)
parser.add_argument(
    "--nki-conv",
    action="store_true",
    help="Route the decoder's 3x3x3 convolutions through the nkilib conv3d kernel, so the "
    "compiler sees one opaque op per convolution instead of expanding 44 of them itself. "
    "This is what the Wan2.2 VAE in this repo does.",
)
parser.add_argument(
    "--nki-min-d-out",
    type=int,
    default=1,
    help="Skip NKI dispatch for convolutions whose input depth is below this. Wan's VAE "
    "uses 2 on measured grounds; this decoder runs at D=1 until the first temporal "
    "upsample, so 1 routes conv_in and the mid blocks too.",
)
parser.add_argument("--skip-device", action="store_true", help="Host reference only")
args = parser.parse_args()


def _build_vae(config, dtype):
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    # Decoder only. The encoder is never called here and its weights are the same order
    # of magnitude as the decoder's, which matters because one logical NeuronCore has
    # 24 GB (96 GB per device / 4 cores) and the staged run hit nrt_tensor_allocate
    # status=4 -- device OOM -- allocating up2's 1.07 GB output.
    vae = AutoencoderKLConv3D.from_config({**config.vae, "only_decoder": True})
    vae = vae.to(dtype).eval()
    # Inference only. Without this the compiled decode traces a backward pass and the
    # Lite backend rejects it with "neuron backend doesn't support events".
    vae.requires_grad_(False)
    vae.use_spatial_tiling = args.tile
    vae.use_temporal_tiling = False
    vae.use_slicing = False
    return vae


def _load_vae_weights(vae, model_path, dtype):
    """Load the checkpoint's ``vae.*`` tensors by name."""
    from safetensors import safe_open

    index = os.path.join(model_path, "model.safetensors.index.json")
    with open(index) as handle:
        weight_map = json.load(handle)["weight_map"]

    targets = {f"vae.{name}": name for name, _ in vae.named_parameters()}
    targets.update({f"vae.{name}": name for name, _ in vae.named_buffers()})
    files: dict[str, list[str]] = {}
    for key in targets:
        if key in weight_map:
            files.setdefault(weight_map[key], []).append(key)

    state: dict[str, torch.Tensor] = {}
    for file_name, keys in files.items():
        with safe_open(os.path.join(model_path, file_name), framework="pt", device="cpu") as f:
            available = set(f.keys())
            for key in keys:
                if key in available:
                    state[targets[key]] = f.get_tensor(key).to(dtype)
    missing = sorted(set(targets.values()) - set(state))
    if missing:
        raise RuntimeError(f"VAE weights missing from checkpoint: {missing[:6]}")
    vae.load_state_dict(state, strict=False, assign=True)
    print(f"loaded {len(state)} VAE tensors")
    return vae


def _count_conv_splits(vae, latents):
    """Run on host with the split path instrumented, so the count is observed not assumed."""
    from vllm_omni.diffusion.models.hunyuan_image3 import autoencoder as ae

    original = ae.Conv3d.forward
    stats = {"calls": 0, "splits": 0, "max_gb": 0.0}

    def counting_forward(self, input):
        b, c, t, h, w = input.shape
        gb = (c * t * h * w) * 2 / 1024**3
        stats["calls"] += 1
        stats["max_gb"] = max(stats["max_gb"], gb)
        if gb > 2:
            stats["splits"] += 1
        return original(self, input)

    ae.Conv3d.forward = counting_forward
    try:
        with torch.no_grad():
            vae.decode(latents, return_dict=False)
    finally:
        ae.Conv3d.forward = original
    return stats


def _relative_error(actual, expected):
    """Max absolute difference, relative to the signal it is measured against."""
    scale = expected.abs().max().item()
    return (actual - expected).abs().max().item() / max(scale, 1e-12)


def _decoder_stages(vae):
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


def _instruction_count(message):
    """Pull the compiler's instruction count out of an NCC_EVRF007/NCC_IXTP002 message."""
    match = re.search(r"[Nn]umber of instructions \((\d+)\)", message) or re.search(
        r"Instructions generated by compiler ([\d,]+)", message
    )
    return int(match.group(1).replace(",", "")) if match else None


def _run_staged(vae, device_vae, latents, dtype, device, reference):
    """Compile and run one graph per stage, mapping cost and feasibility stage by stage.

    A stage that will not compile is reported and skipped: the next stage is fed the host
    activation for that boundary, so a single run characterises every stage instead of
    stopping at the first failure.
    """
    host_stages = _decoder_stages(vae)
    device_stages = _decoder_stages(device_vae)

    # Host pass, keeping each boundary activation as both a timing reference and the
    # input a skipped stage's successor needs.
    boundaries = [latents]
    host_times = []
    with torch.no_grad():
        for _, fn in host_stages:
            start = time.perf_counter()
            boundaries.append(fn(boundaries[-1]))
            host_times.append(time.perf_counter() - start)

    print("\nper-stage, host float32:")
    for (name, _), elapsed, out in zip(host_stages, host_times, boundaries[1:]):
        print(f"  {name:<6} {elapsed:6.2f}s  -> {tuple(out.shape)}")
    print(f"  {'total':<6} {sum(host_times):6.2f}s")

    print("\nper-stage, device:")
    rows = []
    carried = latents.to(dtype).to(device)
    for index, (name, fn) in enumerate(device_stages):
        compiled = torch.compile(
            fn,
            backend=get_compile_backend_name(),
            fullgraph=True,
            dynamic=False,
            options={
                "model_name": f"hunyuan_image3_vae_decode_{name}",
                "compiler_args": _COMPILER_ARGS,
            },
        )
        try:
            start = time.perf_counter()
            out = compiled(carried)
            out.to("cpu")  # execution is queued, so sync before calling the compile done
            compile_seconds = time.perf_counter() - start

            # Device execution is asynchronous: timing each call on its own measures the
            # enqueue, not the work (which is how a stage first reported 0.00s). Run the
            # stage back to back and sync once at the end, so one transfer is amortised
            # over every call instead of being counted in each.
            start = time.perf_counter()
            for _ in range(args.runs):
                last = compiled(carried)
            last.to("cpu")
            best = (time.perf_counter() - start) / args.runs

            # Compare against this stage's own host activation, not just the final
            # image: a single wrong stage is otherwise only visible as a wrong result at
            # the end, with six other suspects.
            expected = boundaries[index + 1]
            got = out.to("cpu").float()
            stage_error = _relative_error(got, expected)
            # Also on the last temporal frame alone: that frame is the image, and maxing
            # over all four hides its error behind the larger early frames -- a 0.263
            # all-frame error on the output stage was a 1.0265 error on the image.
            frame_error = _relative_error(got[:, :, -1:], expected[:, :, -1:])
            print(
                f"  {name:<6} {best:6.2f}s  (host {host_times[index]:.2f}s, "
                f"{host_times[index] / best:.2f}x, compile {compile_seconds:.0f}s) "
                f"err {stage_error:.2e} frame {frame_error:.2e}"
            )
            del expected, got
            rows.append((name, best))
            previous, carried = carried, out
            # Drop the finished stage's graph and its input: both hold device buffers,
            # and nothing downstream reads them again.
            del compiled, previous, last
            gc.collect()
        except Exception as error:  # noqa: BLE001  one failure must not hide the rest
            count = _instruction_count(str(error))
            detail = f"{count:,} instructions" if count else str(error).splitlines()[0][:90]
            print(f"  {name:<6} FAILED   (host {host_times[index]:.2f}s) {detail}")
            rows.append((name, None))
            # Hand the successor the host activation so the map continues past the gap.
            # This transfer can itself fail when the stage died of device OOM, which is
            # how one stage's failure took the whole run down; stop mapping instead.
            try:
                carried = boundaries[index + 1].to(dtype).to(device)
            except Exception as transfer_error:  # noqa: BLE001
                print(f"         cannot continue past it: {transfer_error}")
                break

    ran = [(name, best) for name, best in rows if best is not None]
    print(
        f"\n{len(ran)}/{len(rows)} stages ran on device, "
        f"{sum(best for _, best in ran):.2f}s of device time"
    )
    if len(ran) == len(rows):
        actual = carried.to("cpu").float()
        if actual.shape[-3] != reference.shape[-3]:
            actual = actual[:, :, -reference.shape[-3] :]
        scale = reference.abs().max().item()
        error = (actual - reference).abs().max().item() / max(scale, 1e-12)
        print(f"relative max error vs host float32: {error:.4e} (signal {scale:.4e})")
        return error
    return None


def main() -> None:
    dtype = getattr(torch, args.dtype)
    config = get_config(args.model_path, trust_remote_code=True)
    downsample = config.vae_downsample_factor
    latent_channels = int(config.vae["latent_channels"])
    shape = (
        1,
        latent_channels,
        1,
        args.height // int(downsample[0]),
        args.width // int(downsample[1]),
    )
    print(
        f"latents {shape} -> image {args.height}x{args.width}, device dtype {args.dtype}, "
        f"spatial tiling {'on' if args.tile else 'off'}"
    )

    torch.manual_seed(0)
    latents = torch.randn(shape, dtype=torch.float32)

    host_vae = _load_vae_weights(_build_vae(config, torch.float32), args.model_path, torch.float32)

    if args.report_conv_splits:
        stats = _count_conv_splits(host_vae, latents)
        print(
            f"Conv3d calls {stats['calls']}, >2 GB temporal splits {stats['splits']}, "
            f"largest activation {stats['max_gb']:.2f} GB"
        )
        if stats["splits"]:
            print(
                "  NOTE: the split path mutates self.padding and writes into tensor "
                "slices; expect a fullgraph trace to struggle with it."
            )

    with torch.no_grad():
        reference = host_vae.decode(latents, return_dict=False)[0]
    host_times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        with torch.no_grad():
            host_vae.decode(latents, return_dict=False)
        host_times.append(time.perf_counter() - start)
    host_best = min(host_times)
    print(f"host float32 decode: best {host_best:.2f}s of {args.runs} -> {tuple(reference.shape)}")

    if args.skip_device:
        return

    initialize_lite()
    ensure_current_device_index()
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)

    device_vae = _load_vae_weights(_build_vae(config, dtype), args.model_path, dtype)
    device_vae = device_vae.to(device)

    if args.nki_conv:
        from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_nki_conv import (
            install_nki_conv_dispatch,
        )

        # Install after the move to device: the packed filters have to land on the same
        # device as the weights they came from.
        summary = install_nki_conv_dispatch(
            device_vae, min_d_out=args.nki_min_d_out, verbose=True
        )
        print(f"NKI conv dispatch: {summary['routed']} routed, {summary['skipped']} on compiler")
        for site in summary["sites"]:
            print(f"  {site}")

    if args.stages:
        device_vae.requires_grad_(False)
        with torch.no_grad():
            error = _run_staged(host_vae, device_vae, latents, dtype, device, reference)
        if error is not None and error > args.tolerance:
            raise SystemExit(f"FAIL: {error:.4e} > tolerance {args.tolerance}")
        return

    def decode(z):
        with torch.no_grad():
            return device_vae.decode(z, return_dict=False)[0]

    compiled = torch.compile(
        decode,
        backend=get_compile_backend_name(),
        # Tiling walks several tile shapes, so a graph break per shape is expected.
        fullgraph=not args.tile,
        dynamic=False,
        options={
            "model_name": "hunyuan_image3_vae_decode",
            "compiler_args": _COMPILER_ARGS,
        },
    )

    latents_device = latents.to(dtype).to(device)
    start = time.perf_counter()
    actual = compiled(latents_device).to("cpu").float()
    print(f"device decode, first call (includes compile): {time.perf_counter() - start:.2f}s")

    device_times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        compiled(latents_device).to("cpu")
        device_times.append(time.perf_counter() - start)
    device_best = min(device_times)

    scale = reference.abs().max().item()
    error = (actual - reference).abs().max().item() / max(scale, 1e-12)
    print(f"device decode: best {device_best:.2f}s of {args.runs}")
    print(f"speedup vs host float32: {host_best / device_best:.2f}x")
    print(f"relative max error vs host float32: {error:.4e} (signal {scale:.4e})")
    if error > args.tolerance:
        raise SystemExit(f"FAIL: {error:.4e} > tolerance {args.tolerance}")
    print("OK")


if __name__ == "__main__":
    main()
