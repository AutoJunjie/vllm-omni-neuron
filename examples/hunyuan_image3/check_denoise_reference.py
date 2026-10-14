# SPDX-License-Identifier: Apache-2.0
"""CPU reference for HunyuanImage-3.0 denoise steps (float32, or the device's bfloat16),
through the Neuron pipeline's own code.

The device pipeline runs in bfloat16 with NKI kernels; this runs the same pipeline class
on the host at TP1 in float32 with every kernel on its torch fallback, so the two differ
only in precision and kernels. Prompt templating, RoPE, the generation mask, the prefill
and the denoise math are the pipeline's, not a re-implementation.

Pair it with the step trace (``_StepTrace`` in ``pipeline_hunyuan_image3.py``): a device
run with ``HY3_TRACE_DIR`` records every step's input latents and predictions; this script
then forces those same latents (``HY3_TRACE_FORCE_DIR``) and computes only the requested
steps (``HY3_TRACE_STEPS``), so each device prediction can be compared with the float32
one on identical inputs. The full checkpoint in float32 is ~320 GB of host memory, and a
step costs a few minutes: on the torch fallback every token goes through all 64 experts.

    HY3_TRACE_FORCE_DIR=/workspace/out/trace/hybrid HY3_TRACE_STEPS=0,25,49 \\
    HY3_TRACE_DIR=/workspace/out/trace/cpu_fp32 \\
      python examples/hunyuan_image3/check_denoise_reference.py --seed 42

``--dtype bfloat16`` runs the same CPU steps in the device's dtype instead: the
"expected" side of a three-way comparison (``vllm_neuron.accuracy.testing``), which
separates bfloat16 rounding from what the device adds on top of it.
"""

import argparse
import os
import time
from types import SimpleNamespace

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import torch  # noqa: E402

parser = argparse.ArgumentParser(description="float32 CPU reference denoise steps")
parser.add_argument("--model-path", default="/workspace/models/HunyuanImage-3.0-Instruct")
parser.add_argument(
    "--prompt",
    default=(
        "A cinematic photo of a glass observatory on Mars at sunrise, "
        "volumetric light, ultra detailed"
    ),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image", action="append", default=None, help="A conditioning image (editing)")
parser.add_argument("--guidance", type=float, default=None, help="Default: the checkpoint's (2.5)")
parser.add_argument("--prefill-buckets", default="512", help="Comma-separated prompt buckets")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
args = parser.parse_args()


def _init_single_rank() -> None:
    """One-rank groups the way the engine builds them (vLLM-Omni's parallel_state)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    import vllm_omni_neuron.bootstrap  # noqa: F401

    # The world group first: vLLM-Omni's own path picks a device by rank modulo the
    # device count, which is 0 in CPU mode; given a live world it skips that.
    method = f"tcp://127.0.0.1:{os.environ.get('MASTER_PORT', '29577')}"
    torch.distributed.init_process_group("gloo", init_method=method, world_size=1, rank=0)
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=method, local_rank=0, backend="gloo"
    )
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    _init_single_rank._context = context
    initialize_model_parallel(tensor_parallel_size=1, backend="gloo")


def main() -> None:
    if not os.environ.get("HY3_TRACE_FORCE_DIR") or not os.environ.get("HY3_TRACE_STEPS"):
        raise SystemExit("set HY3_TRACE_FORCE_DIR and HY3_TRACE_STEPS (see the docstring)")
    torch.set_num_threads(os.cpu_count())
    torch.set_default_dtype(torch.float32)
    dtype = getattr(torch, args.dtype)
    _init_single_rank()

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        NeuronHunyuanImage3Pipeline,
    )

    od_config = SimpleNamespace(
        model=args.model_path,
        revision=None,
        # The torch MoE path; host decode (the VAE result is not the point here).
        model_config={
            "moe_kernel": "torch",
            "vae_device": "cpu",
            "prefill_len": 512,
            "prefill_buckets": [int(b) for b in args.prefill_buckets.split(",")],
        },
    )
    start = time.perf_counter()
    with torch.device("cpu"):
        pipeline = NeuronHunyuanImage3Pipeline(od_config)
    pipeline.load_weights()
    # The backbone (and its float32 router), the embedders and the UNet in the requested
    # dtype -- as the device holds them for bfloat16.
    pipeline.model.to(dtype)
    if dtype != torch.float32:
        for layer in pipeline.model.layers:
            layer.mlp.gate_weight.data = layer.mlp.gate_weight.data.float()
    for module in (pipeline.time_embed, pipeline.time_embed_2, pipeline.timestep_emb):
        module.to(dtype)
    pipeline.patch_embed.to(dtype)
    pipeline.final_layer.to(dtype)
    pipeline.wte.to(dtype)
    print(
        f"[ref] pipeline built and loaded in {args.dtype} in {time.perf_counter() - start:.0f}s",
        flush=True,
    )

    # Exactly what forward() builds for a plain request: bot_task auto, no cot text, the
    # checkpoint's guidance and a CPU generator seeded the same way.
    seeds = pipeline.prepare_seed(seed=args.seed, batch_size=1)
    generator = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
    guidance = (
        float(args.guidance)
        if args.guidance is not None
        else float(pipeline.generation_config.diff_guidance_scale)
    )
    batch_cond_image_info = None
    if args.image:
        # The serving path's own pre-processing, then the pipeline's prompt extraction.
        from PIL import Image

        from vllm_omni_neuron.diffusion.models.hunyuan_image3 import (
            get_hunyuan_image_3_pre_process_func,
        )

        request = SimpleNamespace(
            prompt={
                "prompt": args.prompt,
                "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in args.image]},
            },
            sampling_params=SimpleNamespace(width=args.width, height=args.height),
        )
        request = get_hunyuan_image_3_pre_process_func(od_config)(request)
        _, _, _, batch_cond_image_info, _ = pipeline._extract_prompt_inputs(
            [request.prompt], {}, request_id="reference", allow_cond_image=True
        )
    model_inputs = pipeline.prepare_model_inputs(
        prompt=[args.prompt],
        cot_text=None,
        system_prompt=None,
        mode="gen_image",
        generator=generator,
        image_size=(args.height, args.width),
        num_inference_steps=args.steps,
        guidance_scale=guidance,
        bot_task="auto",
        batch_cond_image_info=batch_cond_image_info,
        device=torch.device("cpu"),
    )
    start = time.perf_counter()
    with torch.no_grad():
        pipeline._generate_image(model_inputs, steps=args.steps, guidance=guidance)
    print(
        f"[ref] forced steps {os.environ['HY3_TRACE_STEPS']} done in {time.perf_counter() - start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
