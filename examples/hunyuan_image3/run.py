# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0-Instruct text-to-image on Neuron via the Omni entrypoint.

Usage:
    python examples/hunyuan_image3/run.py --dev                 # 2 steps, warm the graphs
    python examples/hunyuan_image3/run.py                       # 50 steps, 1024x1024
    python examples/hunyuan_image3/run.py --profile --runs 3     # timed warm runs
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
parser.add_argument("--tensor-parallel-size", type=int, default=32)
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
    help="Stage config YAML. Defaults to hunyuan_image3_stage.yaml next to this script.",
)
parser.add_argument(
    "--prefill-len",
    type=int,
    default=None,
    help="Override the prompt bucket. A new value triggers a cold prefill compile.",
)
parser.add_argument(
    "--moe-kernel",
    choices=["nki", "torch"],
    default=None,
    help="MoE MLP implementation. Default comes from the stage config (nki).",
)
parser.add_argument(
    "--platform-target",
    choices=["trn2", "trn3"],
    default=None,
    help="Override Neuron platform detection when the runtime cannot detect it.",
)
parser.add_argument("--profile", action="store_true", help="Run timed warm generations")
parser.add_argument("--runs", type=int, default=1, help="Timed runs when --profile is set")
args = parser.parse_args()

# Env must be set before Omni spawns the diffusion workers.
env_profiles.apply(
    replace(
        env_profiles.HUNYUAN_IMAGE3,
        NEURON_PLATFORM_TARGET_OVERRIDE=args.platform_target,
    ),
    env_profiles.thread_limits(args.tensor_parallel_size),
)


def _load_model_config(stage_cfg_path: str) -> dict:
    with open(stage_cfg_path) as handle:
        stage_cfg = yaml.safe_load(handle)
    return dict(stage_cfg["stage_args"][0]["engine_args"].get("model_config") or {})


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hunyuan_image3_stage.yaml"
    )
    model_config = _load_model_config(stage_cfg)
    if args.prefill_len is not None:
        model_config["prefill_len"] = args.prefill_len
    if args.moe_kernel is not None:
        model_config["moe_kernel"] = args.moe_kernel

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

    print(f"Generating {args.width}x{args.height}, {steps} steps")
    start = time.perf_counter()
    result = omni.generate({"prompt": args.prompt}, params)
    print(f"First (cold) generation: {time.perf_counter() - start:.2f}s")

    if args.profile:
        times = []
        for run in range(args.runs):
            start = time.perf_counter()
            result = omni.generate({"prompt": args.prompt}, params)
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
