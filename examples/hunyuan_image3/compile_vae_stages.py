# SPDX-License-Identifier: Apache-2.0
"""Fill the NEFF cache for the pipeline's on-device VAE decode, all stages in parallel.

With ``vae_device: neuron`` the output rank compiles the decoder's seven stage graphs on
the first request at each resolution. Serially that is ~2 hours cold at 1024x1024
(15 / 101 / 755 / 1489 / 2537 / 2094 / 100 s), all of it inside one request. But a stage
graph depends only on its input shape, not on the activation feeding it, so each stage
can compile in its own process from a zero tensor of the right shape -- the wall time is
the slowest stage (~42 min), and the pipeline then hits every NEFF.

Weights are random: the parameters are graph inputs, so their values never reach the
compiler. Shapes come from a pass over the decoder on the meta device.

    python examples/hunyuan_image3/compile_vae_stages.py                 # all, cores 0-6
    python examples/hunyuan_image3/compile_vae_stages.py --stage up3     # one, this core

Run it from the same checkout the server imports: the cache key includes the source path
and line of every traced op (see ``vae_device``).
"""

import argparse
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description="Precompile the staged device VAE decode")
parser.add_argument("--model-path", default="/workspace/models/HunyuanImage-3.0-Instruct")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--stage", default=None, help="Compile only this stage, on this process")
parser.add_argument(
    "--first-core",
    type=int,
    default=0,
    help="Without --stage: launch one process per stage on consecutive logical cores "
    "starting here. The cores must be free -- stop the server first.",
)
args = parser.parse_args()

STAGES = ["in+mid", "up0", "up1", "up2", "up3", "up4", "out"]


def _launch_all() -> int:
    """One child per stage, each pinned to its own core, then wait for all of them."""
    children = []
    for offset, name in enumerate(STAGES):
        env = dict(os.environ, NEURON_RT_VISIBLE_CORES=str(args.first_core + offset))
        cmd = [sys.executable, os.path.abspath(__file__), "--stage", name]
        cmd += ["--model-path", args.model_path]
        cmd += ["--height", str(args.height), "--width", str(args.width)]
        children.append((name, subprocess.Popen(cmd, env=env)))
        print(f"[launch] {name} on core {args.first_core + offset}", flush=True)
    failed = [name for name, child in children if child.wait() != 0]
    if failed:
        print(f"FAILED: {failed}", flush=True)
        return 1
    print("all stages compiled", flush=True)
    return 0


if args.stage is None:
    sys.exit(_launch_all())
if args.stage not in STAGES:
    raise SystemExit(f"--stage must be one of {STAGES}")

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "600")

import vllm_omni_neuron.bootstrap  # noqa: F401, E402  isort: skip  must precede vllm imports
import torch  # noqa: E402

torch.nn.functional.gelu = torch.ops.aten.gelu.default

from vllm.transformers_utils.config import get_config  # noqa: E402

from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (  # noqa: E402
    StagedDeviceDecoder,
    decoder_stages,
)
from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
    initialize as initialize_lite,
)


def _stage_input_shapes(vae_config, latent_shape):
    """Each stage's input shape, from one pass over a meta-device decoder."""
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    with torch.device("meta"):
        vae = AutoencoderKLConv3D.from_config({**vae_config, "only_decoder": True})
        hidden = torch.empty(latent_shape)
        shapes = {}
        with torch.no_grad():
            for name, fn in decoder_stages(vae):
                shapes[name] = tuple(hidden.shape)
                hidden = fn(hidden)
    return shapes


def main() -> None:
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    config = get_config(args.model_path, trust_remote_code=True)
    vae_config = dict(config.vae)
    factor = int(vae_config["ffactor_spatial"])
    latent_shape = (1, int(vae_config["latent_channels"]), 1)
    latent_shape += (args.height // factor, args.width // factor)
    shape = _stage_input_shapes(vae_config, latent_shape)[args.stage]

    initialize_lite()
    ensure_current_device_index()
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)
    host_vae = AutoencoderKLConv3D.from_config({**vae_config, "only_decoder": True}).eval()
    decoder = StagedDeviceDecoder(host_vae, vae_config, device)
    stage = dict(decoder.stages)[args.stage]

    hidden = torch.zeros(shape, dtype=decoder.dtype).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        stage(hidden).to("cpu")
    print(f"[{args.stage}] input {shape}: compiled and ran in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
