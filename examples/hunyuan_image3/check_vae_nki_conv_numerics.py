# SPDX-License-Identifier: Apache-2.0
"""Is the VAE's NKI conv3d dispatch numerically right, in isolation?

The staged decode runs 11.6x faster on device but produces a wrong image: 1.0265 relative
error on the decoded frame, with per-stage errors of 1e-2 to 2.6e-1 that are too large for
bf16 round-off (~0.4% relative). Either the kernel dispatch is wrong or bf16 is not enough
precision, and the whole-decoder measurement cannot distinguish them -- 38 routed
convolutions, GroupNorm, attention and upsampling all sit in the same number.

So compare one convolution at a time against ``F.conv3d`` on the same device, at the real
decoder shapes, in both dtypes. A unit test cannot do this: the kernel only runs on
hardware, and the CPU-side test of the filter packing is necessarily a round-trip through
the inverse permutation, which cannot detect a layout the *kernel* reads differently.

    python examples/hunyuan_image3/check_vae_nki_conv_numerics.py
    python examples/hunyuan_image3/check_vae_nki_conv_numerics.py --no-lnc-shard
"""

import argparse
import os
import time

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "600")

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
    initialize as initialize_lite,
)

parser = argparse.ArgumentParser(description="NKI conv3d vs torch, at the VAE's shapes")
parser.add_argument(
    "--no-lnc-shard",
    action="store_true",
    help="Trace the kernel without lnc_shard. The decode harness runs on a single logical "
    "NeuronCore, so a kernel that shards across two is the first thing to rule out.",
)
parser.add_argument("--dtypes", default="float32,bfloat16")
parser.add_argument("--tolerance", type=float, default=2e-2)
args = parser.parse_args()

# The shapes that actually run in the decoder: (C_in, C_out, D, H, W). The last is the
# split case -- UpsampleDCAE's 1024 -> 8192, the convolution that needed 4 kernel launches.
SHAPES = [
    (1024, 1024, 1, 64, 64),  # mid.block_1/2
    (1024, 1024, 2, 128, 128),  # up.0 blocks
    (512, 512, 4, 256, 256),  # up.2 blocks
    (128, 128, 4, 1024, 1024),  # up.4 blocks -- top of the kernel's documented range
    (1024, 8192, 1, 64, 64),  # up.0.upsample.conv -- splits into 4 launches
]


def main() -> None:
    initialize_lite()
    ensure_current_device_index()
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)

    if args.no_lnc_shard:
        # Re-trace the kernel without sharding, before anything imports the packed form.
        import nki
        from nkilib.experimental.conv.conv3d import conv3d

        from vllm_omni_neuron.diffusion.models.hunyuan_image3 import vae_nki_conv

        @nki.jit
        def _unsharded(x, filters, bias):
            return conv3d(
                x,
                filters,
                bias,
                stride=(1, 1, 1),
                padding=vae_nki_conv._PADDING,
                dilation=(1, 1, 1),
                lnc_shard=False,
            )

        vae_nki_conv._vae_conv3d_kernel = _unsharded

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_nki_conv import (
        install_nki_conv_dispatch,
    )

    print(f"lnc_shard={'off' if args.no_lnc_shard else 'on'}, device {device}")
    failures = []

    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        print(f"\n{dtype_name}:")
        for in_channels, out_channels, depth, height, width in SHAPES:
            torch.manual_seed(0)
            conv = nn.Conv3d(in_channels, out_channels, 3, padding=1).to(dtype).eval()
            conv.requires_grad_(False)
            x = torch.randn(1, in_channels, depth, height, width, dtype=torch.float32)

            with torch.no_grad():
                reference = conv.float()(x)
            conv = conv.to(dtype).to(device)

            # install_nki_conv_dispatch walks a decoder, so stand one up around this conv.
            holder = nn.Module()
            holder.decoder = nn.Module()
            holder.decoder.add_module("conv", conv)
            summary = install_nki_conv_dispatch(holder, min_d_out=1)
            if not summary["routed"]:
                print(f"  {in_channels:>4}->{out_channels:<4} d{depth} {height}x{width}: not routed")
                continue
            groups = len(conv._nki_packed_filters)

            def run(tensor, module=conv):
                with torch.no_grad():
                    return module(tensor)

            compiled = torch.compile(
                run,
                backend=get_compile_backend_name(),
                fullgraph=True,
                dynamic=False,
                options={
                    "model_name": f"nki_conv_{in_channels}_{out_channels}_{depth}_{height}",
                    "compiler_args": ["--model-type=unet-inference", "-O1"],
                },
            )

            start = time.perf_counter()
            try:
                actual = compiled(x.to(dtype).to(device)).to("cpu").float()
            except Exception as error:  # noqa: BLE001  report and keep going
                print(f"  {in_channels:>4}->{out_channels:<4} d{depth} {height}x{width}: "
                      f"FAILED {str(error).splitlines()[0][:70]}")
                failures.append((dtype_name, in_channels, out_channels, "failed"))
                continue
            elapsed = time.perf_counter() - start

            scale = reference.abs().max().item()
            error = (actual - reference).abs().max().item() / max(scale, 1e-12)
            # A near-zero output is the signature of a kernel that ran but wrote nothing
            # useful, and shows up as error ~= 1 -- worth naming rather than inferring.
            ratio = actual.abs().max().item() / max(scale, 1e-12)
            verdict = "ok" if error <= args.tolerance else "BAD"
            print(
                f"  {in_channels:>4}->{out_channels:<4} d{depth} {height}x{width}: "
                f"err {error:.2e}  out/ref {ratio:.3f}  {groups} launch(es)  "
                f"{elapsed:.0f}s  {verdict}"
            )
            if verdict == "BAD":
                failures.append((dtype_name, in_channels, out_channels, f"{error:.2e}"))

    if failures:
        print(f"\n{len(failures)} shape(s) outside tolerance {args.tolerance}:")
        for dtype_name, in_channels, out_channels, detail in failures:
            print(f"  {dtype_name} {in_channels}->{out_channels}: {detail}")
        raise SystemExit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
