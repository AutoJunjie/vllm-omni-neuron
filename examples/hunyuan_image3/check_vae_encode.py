# SPDX-License-Identifier: Apache-2.0
"""Encode a real image with the device VAE encoder on one core and compare with the host.

The conditioning path of image editing encodes each input image once per request. This
runs the same ``StagedDeviceEncoder`` the pipeline would use, on one logical core, and
reports its time and the posterior moments' error against the host float32 encode.
Precompile its stages in parallel first (``compile_vae_stages.py --encoder``).

    NEURON_RT_VISIBLE_CORES=47 python examples/hunyuan_image3/check_vae_encode.py --image IMG
"""

import argparse
import json
import os
import time

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "300")

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors import safe_open  # noqa: E402
from vllm.transformers_utils.config import get_config  # noqa: E402

from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (  # noqa: E402
    StagedDeviceEncoder,
)
from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
)
from vllm_omni_neuron.lite_compat import (
    initialize as initialize_lite,
)

parser = argparse.ArgumentParser(description="Device VAE encode vs host float32")
parser.add_argument("--model-path", default="/workspace/models/HunyuanImage-3.0-Instruct")
parser.add_argument("--image", required=True)
parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--runs", type=int, default=2)
args = parser.parse_args()


def main() -> None:
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        HunyuanImage3ImageProcessor,
    )

    torch.set_num_threads(os.cpu_count())
    config = get_config(args.model_path, trust_remote_code=True)
    vae_config = dict(config.vae)
    host_vae = AutoencoderKLConv3D.from_config(vae_config).float().eval()
    with open(os.path.join(args.model_path, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    state = {}
    for filename in sorted({v for k, v in weight_map.items() if k.startswith("vae.")}):
        with safe_open(os.path.join(args.model_path, filename), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("vae."):
                    state[key[len("vae.") :]] = f.get_tensor(key).float()
    host_vae.load_state_dict(state, strict=True)

    # The pipeline's own preprocessing: the reso-group resize and the [-1, 1] transform.
    processor = HunyuanImage3ImageProcessor(config)
    image = Image.open(args.image).convert("RGB")
    _, ratio = processor.reso_group.get_base_size_and_ratio_index(*image.size)
    reso = processor.reso_group[int(ratio)]
    pixels = processor.vae_processor(image.resize((int(reso.width), int(reso.height))))
    print(f"pixels {tuple(pixels.shape)}", flush=True)

    start = time.perf_counter()
    with torch.no_grad():
        host = host_vae.encode(pixels.unsqueeze(0).unsqueeze(2)).latent_dist.parameters.float()
    print(f"host float32 encode: {time.perf_counter() - start:.2f}s, moments {tuple(host.shape)}")

    initialize_lite()
    ensure_current_device_index()
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)
    encoder = StagedDeviceEncoder(host_vae, vae_config, device, dtype=getattr(torch, args.dtype))
    for run in range(args.runs):
        start = time.perf_counter()
        moments = encoder.encode(pixels)
        elapsed = time.perf_counter() - start
        half = host.shape[1] // 2
        error = ((moments - host).norm() / host.norm()).item()
        mean_error = ((moments[:, :half] - host[:, :half]).norm() / host[:, :half].norm()).item()
        print(
            f"device encode run {run}: {elapsed:.2f}s, moments relative {error:.3e} "
            f"(mean {mean_error:.3e}), finite {bool(torch.isfinite(moments).all())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
