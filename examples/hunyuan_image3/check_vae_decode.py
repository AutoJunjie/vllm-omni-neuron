# SPDX-License-Identifier: Apache-2.0
"""Can the HunyuanImage-3.0 VAE decoder run on device, and is it faster there?

The decode is the largest remaining term in a warm request — ~26 s of float32 on the
host, 36% of end to end — because this pipeline keeps the VAE off the device. This script
answers both halves of the question before the pipeline is restructured around it: it
loads the real VAE weights, decodes the model's actual latent shape on the host as a
reference, then compiles the same decode for Neuron and reports the error and the speedup.

Two things in this VAE are known risks for a fullgraph trace, and the output says which
one bites:

* its custom ``Conv3d`` splits the temporal axis when an input exceeds 2 GB, and that
  path mutates ``self.padding`` and writes into slices of a freshly padded tensor — the
  same in-graph-mutation shape that made ``neuronx-cc``'s MaskPropagation pass fail for
  the MoE. At 1024x1024 the largest activation should stay under the threshold, so the
  split should never trigger; ``--report-conv-splits`` checks that rather than assuming.
* ``AttnBlock`` and ``UpsampleDCAE`` use einops ``rearrange``.

    python examples/hunyuan_image3/check_vae_decode.py
    python examples/hunyuan_image3/check_vae_decode.py --dtype bfloat16 --height 512
"""

import argparse
import json
import os
import time

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_CC_FLAGS", "-O1 --hbm-scratchpad-page-size=2048")
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
parser.add_argument("--skip-device", action="store_true", help="Host reference only")
args = parser.parse_args()


def _build_vae(config, dtype):
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    vae = AutoencoderKLConv3D.from_config(config.vae)
    vae = vae.to(dtype).eval()
    vae.use_spatial_tiling = False
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
    print(f"latents {shape} -> image {args.height}x{args.width}, device dtype {args.dtype}")

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

    def decode(z):
        return device_vae.decode(z, return_dict=False)[0]

    compiled = torch.compile(
        decode,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": "hunyuan_image3_vae_decode",
            "compiler_args": [
                "--model-type=unet-inference",
                "--auto-cast=none",
                "--internal-max-instruction-limit=15000000",
                "-O1",
                "--hbm-scratchpad-page-size=2048",
            ],
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
