# SPDX-License-Identifier: Apache-2.0
"""The tier-2 reference: denoising steps of a recorded request by the official implementation.

Replays a device trace (``HY3_TRACE_DIR``: ``inputs.pt`` plus ``step_NN.pt``) through
the checkpoint's own modeling code on the host (:mod:`hy3_official`), the whole 80B model
in ``--dtype``, and writes the same trace fields for each step:

- the request's templated inputs (token ids, image slots, attention mask) come from the
  trace -- the plugin builds them with vLLM-Omni's port of the official template; RoPE,
  the token embeddings, every layer, the patch embed / final layer, the CFG combine and
  the scheduler step are the official code's;
- each step starts from the trace's input latents and timestep (teacher forcing), with
  ``first_step=True`` -- the whole sequence recomputed, no KV cache -- so a step needs no
  earlier one.

Text-to-image requests only (the RoPE layout below is the one generated image).

    python test/neuron/hy3_official_step.py --model-path M --trace DIR --steps 0 \\
        --dtype float32 --output OUT
"""

import argparse
import os
import sys

import torch
from hy3_official import build, load, official_module

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--trace", required=True, help="the device's HY3_TRACE_DIR")
parser.add_argument("--steps", default="0")
parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
parser.add_argument("--threads", type=int, default=os.cpu_count())
parser.add_argument("--layers", type=int, default=None, help="smoke test: only the first N layers")
parser.add_argument("--output", required=True)
args = parser.parse_args()


def main() -> None:
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    module, config = official_module(args.model_path)
    if args.layers:
        config.num_hidden_layers = args.layers
    model = load(build(lambda: module.HunyuanImage3ForCausalMM(config), dtype), args.model_path, "", strict=False)
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig.from_pretrained(args.model_path)
    model.cached_rope = module.CachedRoPE(config)  # built on the meta device above

    inputs = torch.load(os.path.join(args.trace, "inputs.pt"), weights_only=False)
    if inputs.get("cond_vae_images") is not None:
        raise SystemExit("hy3_official_step.py replays text-to-image requests only")
    input_ids, image_mask = inputs["input_ids"], inputs["image_mask"]
    cfg_factor = input_ids.shape[0]
    columns = image_mask[0].nonzero().flatten()
    start, stop = int(columns[0]), int(columns[-1]) + 1
    first = torch.load(os.path.join(args.trace, "step_00.pt"))
    token_h, token_w = first["latents_in"].shape[-2:]  # patch size 1: a token per latent pixel
    rope_image_info = [[(slice(start, stop), (int(token_h), int(token_w)))]] * cfg_factor
    # What generate_image sets before its pipeline runs (read by forward, used only on
    # steps after the first): one generated image, one timestep token before it.
    model.num_image_tokens, model.num_special_tokens = int(token_h) * int(token_w), 1
    model.post_token_len = int(input_ids.shape[1] - stop)

    # The official pipeline's scheduler and CFG operator (HunyuanImage3ForCausalMM.pipeline).
    scheduler = module.FlowMatchDiscreteScheduler(
        shift=model.generation_config.flow_shift, reverse=True, solver="euler"
    )
    scheduler.set_timesteps(int(inputs["steps"]))
    cfg = sys.modules[module.HunyuanImage3Text2ImagePipeline.__module__].ClassifierFreeGuidance()
    os.makedirs(args.output, exist_ok=True)
    for step in (int(s) for s in args.steps.split(",")):
        recorded = torch.load(os.path.join(args.trace, f"step_{step:02d}.pt"))
        timestep = scheduler.timesteps[step]
        if not torch.allclose(timestep, recorded["timestep"].float().reshape(()), atol=1e-3):
            raise SystemExit(f"step {step}: official timestep {timestep} vs recorded {recorded['timestep']}")
        latents = recorded["latents_in"]
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                rope_image_info=rope_image_info,
                images=torch.cat([latents] * cfg_factor).to(dtype),
                image_mask=image_mask,
                timesteps=timestep.repeat(cfg_factor),
                timesteps_index=inputs["gen_timestep_scatter_index"],
                gen_timestep_scatter_index=inputs["gen_timestep_scatter_index"],
                mode="gen_image",
                first_step=True,
                use_cache=False,
            )
        raw = out["diffusion_prediction"].float()
        pred = cfg(*raw.chunk(2), float(inputs["guidance"]), step=step) if cfg_factor == 2 else raw
        scheduler._step_index = step
        latents_out = scheduler.step(pred, timestep, latents, return_dict=False)[0]
        torch.save(
            {"timestep": recorded["timestep"], "latents_in": latents, "pred_raw": raw, "pred": pred,
             "latents_out": latents_out},
            os.path.join(args.output, f"step_{step:02d}.pt"),
        )
        print(f"step {step}: done", flush=True)


if __name__ == "__main__":
    main()
