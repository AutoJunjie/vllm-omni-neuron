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
| `moe_cte` (blockwise, `shard_on_block`) | routed experts | every local expert, densely |

Each kernel is gated on `can_run_kernel` and its own tiling limits, so CPU mode and
fake-tensor tracing take the torch path.

The MoE kernel is on by default (`model_config.moe_kernel: nki`, `moe_block_size: 1024`);
`torch` switches the routed experts back to dense matmuls, which is the simpler path for
bisecting. Two kernels were evaluated for the MoE and only one can even express this
model's step:

- nkilib's dense **`mlp`** (SwiGLU) is numerically correct — it is what confirmed that
  nkilib applies the activation to the *gate* operand — but it validates its own tile
  budget at trace time and rejects anything above **256 tokens per launch**
  (`[NCC_INKI016] Stack out of memory`). The limit is on `batch * seq`, so folding the
  tokens into a wider, shorter batch does not help. At 8194 tokens that is 33 launches
  per expert, more overhead than the matmuls it would replace.
- **`moe_cte`** blocks internally at `block_size`, so one call per layer covers the whole
  step. It consumes the router's dense `[T, E_local]` affinities directly and computes
  only the pairs the router selected.

Getting `moe_cte` to compile inside the model took one non-obvious fix. On its own it was
clean at every token count the model uses, with up to 32 chained instances and distinct
weights, with the router in-graph, and with either MoE process group — yet a 1-layer
model ICEd `neuronx-cc` with
`[NCC_IMPR902] MaskPropagation error: call to isl_set_union failed: spaces don't match`.
The difference was how the padding mask reached the kernel: the standalone check passed
it in as a graph input, while the model built it in-graph with `torch.zeros` plus a slice
assignment. Replacing that write with an `arange < live` comparison — and omitting the
mask entirely when the token count already fills whole blocks — compiles. A mask built by
mutation is what the failing pass could not handle.

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
CFG batch of 2 (`guidance_scale` 2.5).

Steady state, from `pipeline_perf_metrics.json` on the third warm generation:

| Stage | Seconds |
| --- | --- |
| Prompt prefill | 0.05 |
| Denoise (50 steps) | 44.75 (**0.895 / step**) |
| VAE decode (host, float32) | 27.8 |
| End to end | 72.7 |

Two independent benchmarks agree:

- **This repo's own runner**, `examples/hunyuan_image3/run.py --profile --runs 3` (one
  cold generation to build the NEFFs, then timed warm generations): **72.83 s** average,
  72.53 s min, 73.24 s max. Its per-stage metrics are the table above.
- **vLLM-Omni's `diffusion_benchmark_serving.py`** over 4 sequential requests against the
  served endpoint: 4/4 successful, 282.55 s wall, latency mean **70.64 s**, median
  70.68 s, P95 71.01 s, P99 71.04 s, `stage_0_gen_ms` mean 70105. The ~2 s gap is the
  offline runner's per-request Python setup, which the server amortizes.

The first request on a cold NEFF cache is much slower because `torch.compile` builds both
graphs on their first call, inside the generation: 1963 s end to end, of which roughly
30 minutes is compilation. Point `TORCH_NEURONX_NEFF_CACHE_DIR` at storage that outlives
the container and only the first run ever pays it; a warm server then becomes healthy in
about 90 seconds and its first request is already at steady state.

Output is **byte-identical** (`sha256 f62b9113...` for the Mars-observatory prompt at
seed 42) across all three axes that could have perturbed it: the offline runner vs the
OpenAI chat endpoint, a cold NEFF build vs a warm cache, and two different physical
`trn2.48xlarge` instances. So neither the serving path nor the compilation cache nor the
host introduces numerical divergence.

The routed MoE is the dominant term, and the two ways to compute it land in the same
place once the kernel's block size is tuned:

| Routed MoE | s/step (denoise) | Warm end to end (3 runs) |
| --- | --- | --- |
| blockwise NKI, `moe_block_size: 1024` (default) | 0.911 | 72.10 s (71.95 / 72.27) |
| dense matmuls (`moe_kernel: torch`) | **0.895** | 72.83 s (72.53 / 73.24) |

Block size matters much more than the arithmetic does. Sweeping it at 1024x1024 / 50
steps, s/step of denoise:

| `moe_block_size` | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| s/step | 1.115 | 0.911 | 0.911 | 0.949 |
| warm end to end | 83.00 s | 74.05 s | 72.10 s | 76.17 s |

Only 12.7% of the local (token, expert) pairs are live, so small blocks spend most of
their time on per-block bookkeeping — the token permutation plus
`build_blockwise_mapping`'s own NKI kernels, 32 times per step — while large ones pad
dead rows. At 1024 the blockwise path is at **parity** with dense: its denoise phase is
still 1.8% slower, and the end-to-end ordering flips only because the host VAE decode
varies by ~2.7 s run to run (25.1-27.8 s observed). So the honest read is that NKI covers
the dominant compute at no measured cost, not that sparsity wins here — the ~8x FLOP
saving is spent entirely on block bookkeeping.

That leaves the host-side VAE decode as the largest remaining term, at 36% of a warm
request.

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

Its measured relative error against the reference forward is 7.07e-08 — float32
round-off, so the ported math is exact.

`test/unit/test_hunyuan_image3_sharding.py` covers what TP1 cannot: it drives each
weight loader rank by rank at TP16 and TP32 and asserts the shards tile the checkpoint in
the order the forward assumes, including that each rank's query heads belong to the KV
head it loaded (the replicated-KV grouping, which only matters when
`num_key_value_heads < tp_size`).

`examples/hunyuan_image3/check_nki_kernels.py` compiles each NKI kernel on its own, at
the shapes the real model runs, and diffs it against the torch fallback. On device at
TP32: `output_projection_cte` 1.96e-03 relative (BF16-level), `attention_cte` passing at
the same tolerance.

On device, end to end: a 1024x1024 50-step generation of
`"A cinematic photo of a glass observatory on Mars at sunrise, volumetric light, ultra
detailed"` at seed 42 produces a coherent image, through both the offline runner and the
OpenAI chat endpoint, with byte-identical output. There is **no** image-quality benchmark
score (no GEBench or equivalent) and no per-step comparison against a GPU run of the same
checkpoint, so the correctness claim is "reference math reproduced, and the full pipeline
generates what the prompt asks for" — not quality parity with the reference deployment.

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
- Sparsity buys nothing here. The blockwise MoE kernel is the default and generates a
  coherent image over all 32 layers, but at best it matches the dense matmuls rather than
  beating them: block bookkeeping consumes the whole ~8x arithmetic saving (see
  [Measured performance](#measured-performance)). Making that bookkeeping cheaper — or
  a kernel that fuses the routing metadata into the expert matmul — is where a real win
  would come from.
- An earlier report that the dense MLP kernel "wedged a NeuronCore" was wrong. It raises
  a clean compile-time validation error; the node losses that coincided with it have a
  separate explanation (a standing health-agent flag plus node auto-recovery).

## Related information

- [Onboarding a model](../model-dev/onboarding-models.md) — the process this
  implementation follows.
- [Setup guide](../getting-started/setup-guide.md) — environment and container.
- [Wan2.2-T2V-A14B model card](wan22-t2v-14b.md) — the reference blueprint.
