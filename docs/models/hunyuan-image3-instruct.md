# HunyuanImage-3.0-Instruct

<!-- meta: description: Model card for HunyuanImage-3.0-Instruct text-to-image
generation on AWS Trainium with the vLLM Omni Neuron plugin — supported
configurations, the two-graph execution model, NKI kernels, and known limits. -->
<!-- meta: keywords: vLLM Omni, Neuron, Trainium, trn2, HunyuanImage-3.0,
HunyuanImage3ForCausalMM, text-to-image, DiT, MoE, expert parallelism, NKI,
torch.compile, libtorch-neuronx-lite -->
<!-- meta: content_type: reference -->
<!-- meta: date_updated: 2026-10-09 -->

## Overview

[`tencent/HunyuanImage-3.0-Instruct`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct)
is an 80B-parameter mixture-of-experts diffusion transformer that generates images
from text. This plugin serves its **DiT stage only**: the prompt goes straight to the
diffusion stage, with no autoregressive recaption stage in front of it. That is the
same scope as upstream vLLM-Omni's `vllm_omni/deploy/hunyuan_image3_dit.yaml`.

Registered architecture key: `HunyuanImage3ForCausalMM` — the Neuron pipeline replaces
upstream's `HunyuanImage3Pipeline` for that key, so a stage config only needs
`model_class_name: HunyuanImage3ForCausalMM`.

## Architecture

| Property | Value |
| --- | --- |
| Decoder layers | 32 |
| Hidden size | 4096 |
| Attention | 32 query heads / 8 KV heads, head dim 128, QK RMSNorm |
| Position encoding | 2D RoPE (theta 10000), applied before the QK norm |
| Routed experts | 64, top-8, renormalised, intermediate 3072 |
| Shared expert | 1, intermediate 3072 |
| Activation | SwiGLU (`down(chunk0 * silu(chunk1))` over the fused `gate_and_up_proj`) |
| Latents | 32 channels, VAE spatial factor 16 (1024x1024 -> 64x64 -> 4096 tokens) |
| Scheduler | FlowMatchEulerDiscrete, shift 3.0 |
| Checkpoint dtype | BF16 |

## Supported configuration

| Setting | Value |
| --- | --- |
| Instance | one `trn2.48xlarge` |
| Stage config | [`examples/hunyuan_image3/hunyuan_image3_stage.yaml`](../../examples/hunyuan_image3/hunyuan_image3_stage.yaml) (TP32), [`..._tp16.yaml`](../../examples/hunyuan_image3/hunyuan_image3_stage_tp16.yaml) (TP16) |
| Parallelism | tensor/expert parallel only — no CP, no CFG parallelism, no VAE patch parallelism |
| Resolution | 1024x1024 (the compiled denoise graph is shape-specialised) |
| Batch | one request at a time (`max_batch_size: 1`) |
| Quantization | BF16 only |
| Task | text-to-image; conditioning images (image edit / IT2I) are rejected |

At TP32 each rank owns 1 query head (each of the 8 KV heads is replicated across its 4
query ranks) and 2 of the 64 routed experts, which puts roughly 5 GB of weights on each
of the 24 GB logical cores. TP16 doubles both (2 query heads, 4 experts, ~10 GB).

## Execution model

Upstream runs the first denoising step over `prompt + image tokens` and later steps over
`timestep + image tokens`, reusing a prompt KV cache. Two different shapes means two
graphs on Neuron, so the plugin makes that split explicit and compiles each once:

1. **Prefill** — `forward_prefill(inputs_embeds, cos, sin)` runs causal self-attention
   over the prompt and returns the per-layer prompt K/V. Once per request.
2. **Denoise step** — one graph that fuses the patch embedder (`UNetDown`), the 32-layer
   backbone and the final layer (`UNetUp`), consuming the prompt K/V plus the step's own
   image K/V. Replayed every scheduler step.

The split is exact rather than an approximation. HunyuanImage3's generation mask is
lower-triangular with a full-attention block over the generated-image span, so prompt
tokens never attend to image tokens, and generated tokens attend to the whole prompt.
The prompt is right-padded to the compiled `prefill_len` bucket — safe under causal
attention — and those columns are masked out of every denoise step by additive key
biases read straight out of that same generation mask. The timestep token and the image
tokens get one bias each, because the mask gives them different visibility.

The scheduler, the CFG combine, the 133k x 4096 token embedding and the VAE decode stay
on the host. That keeps a 1.1 GB table off every core, and means no uncompiled operation
is ever dispatched to the device under the Lite runtime.

### NKI kernels

| Kernel | Where | Fallback |
| --- | --- | --- |
| `attention_cte` (causal, d-major output) | prompt prefill attention | torch masked softmax |
| `output_projection_cte` | every attention output projection | `matmul` + bias |
| nkilib `mlp` (SwiGLU, `ActFnType.SiLU`) | every routed and shared expert | `silu(x @ gate) * (x @ up)` then `@ down` |

Each kernel is gated on `can_run_kernel` and its own tiling limits, so CPU mode and
fake-tensor tracing take the torch path. `model_config.moe_kernel: torch` forces the
fallback for the MLPs, which is the first thing to try when bisecting an accuracy
regression.

The denoise step's attention stays in torch on purpose: `attention_cte` takes no
per-position mask, and that attention is around 1.5% of a layer's FLOPs next to the
64-expert MoE.

## Usage

Offline:

```bash
python examples/hunyuan_image3/run.py \
  --model-path /models/HunyuanImage-3.0-Instruct \
  --output hunyuan_image3_output.png
```

Online (OpenAI-compatible chat API):

```bash
bash examples/hunyuan_image3/serve.sh /models/HunyuanImage-3.0-Instruct 8091
```

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"a red teapot on a wooden table"}],
       "extra_body":{"height":1024,"width":1024,"num_inference_steps":50,
                     "guidance_scale":2.5,"seed":42}}' \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- | base64 -d > hunyuan_image3_output.png
```

Per-request stage timings are written to
`$WORKLOAD_OUTPUT_RW/metrics/pipeline_perf_metrics.json` (`/tmp/metrics/...` by default):
`prefill_seconds`, `denoise_seconds`, `seconds_per_step`, `vae_decode_seconds` and
`e2e_forward_seconds`.

## Measured performance

One `trn2.48xlarge`, TP32, 1024x1024, 50 denoising steps, BF16, `moe_kernel: torch`,
CFG batch of 2 (`guidance_scale` 2.5), warm NEFF cache:

| Stage | Seconds |
| --- | --- |
| Prompt prefill | 23.7 |
| Denoise (50 steps) | 1911.4 (38.2 / step) |
| VAE decode (host, float32) | 28.1 |
| End to end | 1963.3 |

The cold NEFF build for both graphs takes roughly 30 minutes on top of that; a warm
server answers its first request immediately and becomes healthy in about 90 seconds.

The offline runner and the OpenAI chat endpoint produce **byte-identical** output for the
same prompt, seed and step count (`sha256 f62b9113...` for the Mars-observatory prompt at
seed 42), so the serving path adds no numerical divergence.

Per-step time is dominated by the routed MoE: every rank evaluates both of its local
experts densely for all 8194 tokens, which is about 8x the FLOPs that top-8-of-64 needs,
and it runs as plain matmuls because the nkilib SwiGLU kernel is disabled (see
[Known limits](#known-limits)). Sparse expert dispatch and a working MLP kernel are the
two levers worth pulling first.

## Validation

`test/unit/test_hunyuan_image3_numerics.py` builds a tiny random checkpoint in the real
HunyuanImage-3.0 key layout and asserts that the prefill + denoise split reproduces a
single full-sequence forward of the reference math from
`modeling_hunyuan_image_3.py`. It covers the fused `qkv_proj` row grouping, the
`gate_and_up_proj` chunk order, RoPE-before-QK-norm ordering, the float32 router, and
the attention-mask decomposition. It runs on CPU at TP1:

```bash
VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_numerics.py
```

## Known limits

- One resolution per compiled graph. Changing height/width, `prefill_len`, the step
  layout or the TP degree triggers a cold NEFF build; point
  `TORCH_NEURONX_NEFF_CACHE_DIR` at storage that outlives the container.
- A prompt longer than `prefill_len` is an error, not a silent truncation. Raise the
  bucket (and pay one recompile).
- No CFG parallelism yet: both CFG branches run in the same batch, so a guided request
  costs twice the tokens per step. `guidance_scale <= 1.0` runs a single branch.
- No conditioning-image (IT2I / image edit) path, no sequence/context parallelism, no
  FP8.
- The VAE decode runs on the host in float32 on the output rank only, so it is neither
  accelerated nor parallel (28 s at 1024x1024).
- The nkilib SwiGLU MLP kernel is not validated on this stack. On a managed cluster a
  hung NeuronCore can get the whole instance replaced within seconds, so set
  `NEURON_RT_EXEC_TIMEOUT` (the env profile defaults it to 600 s) and disable node
  auto-recovery before re-testing it.

## Related information

- [Onboarding a model](../model-dev/onboarding-models.md) — the process this
  implementation follows.
- [Setup guide](../getting-started/setup-guide.md) — environment and container.
- [Wan2.2-T2V-A14B model card](wan22-t2v-14b.md) — the reference blueprint.
