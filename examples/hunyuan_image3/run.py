# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0-Instruct on Neuron via the Omni entrypoint.

One 1024x1024 image per prompt, with classifier-free guidance: straight from the prompt
(the stage config's ``bot_task: image``), after think / recaption text generated on the
backbone (``--bot-task``), or edited from conditioning images (``--image``).

Usage:
    python examples/hunyuan_image3/run.py --dev                 # 2 steps, warm the graphs
    python examples/hunyuan_image3/run.py                       # 50 steps, 1024x1024
    python examples/hunyuan_image3/run.py --profile --runs 3     # timed warm runs
    python examples/hunyuan_image3/run.py --bot-task think \
        --stage-config examples/hunyuan_image3/hunyuan_image3_dit_ar.yaml
    python examples/hunyuan_image3/run.py --image input.png \
        --prompt "Turn the sky into a starry night, keep everything else unchanged."
"""

import argparse
import os
import time
from dataclasses import replace

import torch
import yaml

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

# torch_neuronx / Lite patch F.gelu with a wrapper around the C builtin, which Dynamo
# cannot trace in fullgraph mode. Restore the dispatcher-aware aten op.
torch.nn.functional.gelu = torch.ops.aten.gelu.default

parser = argparse.ArgumentParser(description="HunyuanImage-3.0-Instruct on Neuron")
parser.add_argument("--dev", action="store_true", help="Dev mode: 2 denoising steps")
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=None,
    help="The stage's total ranks (TP x CFG x CP), checked against the stage config; "
    "the thread caps are sized from the stage config either way.",
)
parser.add_argument(
    "--model-path",
    type=str,
    default="tencent/HunyuanImage-3.0-Instruct",
    help="Local checkpoint directory, or a Hugging Face repo id to download.",
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--num-inference-steps", type=int, default=None)
parser.add_argument(
    "--guidance-scale",
    type=float,
    default=None,
    help="Defaults to the checkpoint's generation_config (2.5).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--prompt",
    type=str,
    default=(
        "A cinematic photo of a glass observatory on Mars at sunrise, "
        "volumetric light, ultra detailed"
    ),
)
parser.add_argument("--output", type=str, default="hunyuan_image3_output.png")
parser.add_argument(
    "--stage-config",
    type=str,
    default=None,
    help="Stage config YAML. Defaults to hunyuan_image3_dit.yaml next to this script.",
)
parser.add_argument(
    "--prefill-len",
    type=int,
    default=None,
    help="Override the base prompt bucket (always one of the buckets, next to the stage "
    "config's prefill_buckets). A new value triggers a cold prefill and denoise compile.",
)
parser.add_argument(
    "--moe-kernel",
    choices=["nki", "torch"],
    default=None,
    help="MoE MLP implementation. Default comes from the stage config (nki).",
)
parser.add_argument(
    "--bot-task",
    choices=["think", "recaption", "think_recaption"],
    default=None,
    help="Generate think / recaption text on the backbone before the image (AR + DiT). "
    "Needs a stage config with a prefill bucket that fits the system prompt plus the "
    "generated text, e.g. hunyuan_image3_dit_ar.yaml.",
)
parser.add_argument(
    "--vae-device",
    choices=["neuron", "cpu"],
    default=None,
    help="Where the VAE decodes. Default comes from the stage config (neuron).",
)
parser.add_argument(
    "--platform-target",
    choices=["trn2", "trn3"],
    default=None,
    help="Override Neuron platform detection when the runtime cannot detect it.",
)
parser.add_argument(
    "--vae-parallel",
    type=int,
    default=None,
    help="Split the device VAE decode by rows over this many ranks (2 in the stage config; "
    "1 = the whole decode on the output rank's core).",
)
parser.add_argument(
    "--prefill-buckets",
    default=None,
    help="Comma-separated prompt buckets, e.g. 512,6144; each request takes the smallest "
    "that fits (each bucket is its own prefill and denoise compile).",
)
parser.add_argument(
    "--image",
    action="append",
    default=None,
    help="A conditioning image (image editing / IT2I); repeat for several. The prompt is the "
    "instruction. Needs a prompt bucket that fits the image tokens (prefill_buckets).",
)
parser.add_argument(
    "--num-layers",
    type=int,
    default=None,
    help="Bring-up only: truncate the decoder stack (the image is meaningless).",
)
parser.add_argument("--profile", action="store_true", help="Run timed warm generations")
parser.add_argument("--runs", type=int, default=1, help="Timed runs when --profile is set")
args = parser.parse_args()

def _stage_config_path() -> str:
    return args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hunyuan_image3_dit.yaml"
    )


def _stage_world_size(stage_cfg_path: str) -> int:
    """The stage's ranks: tensor_parallel_size x cfg_parallel_size x ring_degree."""
    with open(stage_cfg_path) as handle:
        parallel = yaml.safe_load(handle)["stage_args"][0]["engine_args"]["parallel_config"]
    return (
        int(parallel["tensor_parallel_size"])
        * int(parallel.get("cfg_parallel_size") or 1)
        * int(parallel.get("ring_degree") or 1)
    )


# Env must be set before Omni spawns the diffusion workers.
env_profiles.apply(
    replace(
        env_profiles.HUNYUAN_IMAGE3,
        NEURON_PLATFORM_TARGET_OVERRIDE=args.platform_target,
    ),
    env_profiles.thread_limits(_stage_world_size(_stage_config_path())),
)


def _load_stage_args(stage_cfg_path: str) -> dict:
    with open(stage_cfg_path) as handle:
        return yaml.safe_load(handle)["stage_args"][0]


def _load_model_config(stage_cfg_path: str) -> dict:
    return dict(_load_stage_args(stage_cfg_path)["engine_args"].get("model_config") or {})


def _check_tensor_parallel_size(stage_cfg_path: str) -> None:
    """--tensor-parallel-size, when given, must be the stage's rank count."""
    world = _stage_world_size(stage_cfg_path)
    if args.tensor_parallel_size is not None and args.tensor_parallel_size != world:
        raise SystemExit(
            f"--tensor-parallel-size={args.tensor_parallel_size} disagrees with the "
            f"{world} ranks of {stage_cfg_path} (tensor_parallel_size x cfg_parallel_size "
            "x ring_degree). The stage config decides the mesh and the visible core range."
        )


def _write_stage_config_with(stage_cfg_path: str, overrides: dict) -> str:
    """A temporary copy of the stage config with ``overrides`` merged into model_config."""
    import tempfile

    with open(stage_cfg_path) as handle:
        config = yaml.safe_load(handle)
    engine_args = config["stage_args"][0]["engine_args"]
    engine_args["model_config"] = {**(engine_args.get("model_config") or {}), **overrides}
    handle = tempfile.NamedTemporaryFile(
        "w", prefix="hunyuan_image3_stage_", suffix=".yaml", delete=False
    )
    with handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return handle.name


def main() -> None:
    stage_cfg = _stage_config_path()
    _check_tensor_parallel_size(stage_cfg)
    overrides = {
        key: value
        for key, value in (
            ("prefill_len", args.prefill_len),
            ("moe_kernel", args.moe_kernel),
            ("vae_device", args.vae_device),
            ("num_layers", args.num_layers),
            ("vae_parallel", args.vae_parallel),
            (
                "prefill_buckets",
                [int(b) for b in args.prefill_buckets.split(",")] if args.prefill_buckets else None,
            ),
        )
        if value is not None
    }
    if overrides:
        # Omni's model_config argument did not reach the engine (--moe-kernel torch still
        # ran the NKI MoE), while the stage config always does, so overrides go into a copy.
        stage_cfg = _write_stage_config_with(stage_cfg, overrides)
        print(f"[init] model_config overrides {overrides} -> {stage_cfg}")
    model_config = _load_model_config(stage_cfg)

    # A cold compile of the 80B MoE denoise graph runs far past vllm-omni's 600 s
    # worker handshake, so raise the ceiling before Omni starts the orchestrator.
    cold_timeout = int(os.environ.get("HUNYUAN_HANDSHAKE_TIMEOUT_S", "14400"))
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as stage_proc

        if getattr(stage_proc, "_HANDSHAKE_POLL_TIMEOUT_S", 0) < cold_timeout:
            stage_proc._HANDSHAKE_POLL_TIMEOUT_S = cold_timeout
            print(f"[init] raised diffusion handshake timeout to {cold_timeout}s")
    except Exception as error:  # pragma: no cover - best effort
        print(f"[init] could not patch handshake timeout ({error!r}); using the default")

    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=cold_timeout,
        init_timeout=cold_timeout,
        model_config=model_config,
    )

    steps = args.num_inference_steps or (2 if args.dev else 50)
    sampling_kwargs = dict(
        height=args.height,
        width=args.width,
        num_inference_steps=steps,
        seed=args.seed,
    )
    if args.guidance_scale is not None:
        # guidance_scale alone is ignored downstream; the pipeline reads the
        # `_provided` flag to tell an explicit value from the schema default.
        sampling_kwargs.update(
            guidance_scale=args.guidance_scale, guidance_scale_provided=True
        )
    params = OmniDiffusionSamplingParams(**sampling_kwargs)

    request = {"prompt": args.prompt}
    if args.image:
        from PIL import Image

        request["multi_modal_data"] = {
            "image": [Image.open(path).convert("RGB") for path in args.image]
        }
    if args.bot_task is not None:
        request["bot_task"] = args.bot_task
    print(f"Generating {args.width}x{args.height}, {steps} steps, bot_task={args.bot_task}")
    start = time.perf_counter()
    result = omni.generate(request, params)
    print(f"First (cold) generation: {time.perf_counter() - start:.2f}s")

    if args.profile:
        times = []
        for run in range(args.runs):
            start = time.perf_counter()
            result = omni.generate(request, params)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"[profile] run {run + 1}/{args.runs}: {elapsed:.2f}s")
        print("=" * 60)
        print(f"[profile] {args.width}x{args.height}, {steps} steps, {args.runs} run(s)")
        print(f"  avg {sum(times) / len(times):.2f}s  min {min(times):.2f}s  max {max(times):.2f}s")
        print("=" * 60)

    images = result[0].request_output.images
    if not images:
        raise RuntimeError("No image returned by the Omni engine")
    image = images[0]
    if not hasattr(image, "save"):
        from diffusers.image_processor import VaeImageProcessor

        image = VaeImageProcessor.numpy_to_pil(image)[0]
    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
