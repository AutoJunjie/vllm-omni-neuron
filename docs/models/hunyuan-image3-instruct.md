# HunyuanImage-3.0-Instruct

<!-- meta: description: Model card for HunyuanImage-3.0-Instruct text-to-image
generation on AWS Trainium with the vLLM Omni Neuron plugin — supported
configurations, the two-graph execution model, NKI kernels, and known limits. -->
<!-- meta: keywords: vLLM Omni, Neuron, Trainium, trn2, HunyuanImage-3.0,
HunyuanImage3ForCausalMM, text-to-image, DiT, MoE, expert parallelism, NKI,
torch.compile, libtorch-neuronx-lite -->
<!-- meta: content_type: reference -->
<!-- meta: date_updated: 2026-10-12 -->

## Overview

[`tencent/HunyuanImage-3.0-Instruct`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct)
is an 80B-parameter mixture-of-experts diffusion transformer that generates images
from text. By default the prompt goes straight to the diffusion stage, the same scope as
upstream vLLM-Omni's `vllm_omni/deploy/hunyuan_image3_dit.yaml`. A request with
`bot_task` `think`, `recaption` or `think_recaption` first generates that text on the
same backbone and then conditions the image on it -- see
[Think / recaption (AR + DiT)](#think--recaption-ar--dit).

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
| Parallelism | TP16 x CP2 by default (`context_parallel: 2`): tensor/expert parallel over each half of the ranks; with CFG each half runs one CFG branch (`cfg_parallel`, default on), without it the denoise tokens are split across the halves. `context_parallel: 1` is TP32. No VAE patch parallelism. The UNet patch embed and final layer are split across all ranks by GroupNorm group (`unet_parallel`, default on). See [Context parallelism](#context-parallelism) |
| Step caching | Cache-DiT DBCache over the denoise loop, `residual_diff_threshold` 0.12 by default (`step_cache`); vllm-omni's TeaCache for this model as an alternative (`tea_cache`, off). See [Step caching](#step-caching) |
| Resolution | 1024x1024 (the compiled denoise graph is shape-specialised) |
| Batch | one request at a time (`max_batch_size: 1`) |
| Quantization | BF16 only |
| Task | text-to-image; conditioning images (image edit / IT2I) are rejected |

At TP32 each rank owns 1 query head (each of the 8 KV heads is replicated across its 4
query ranks) and 2 of the 64 routed experts, which puts roughly 5 GB of weights on each
of the 24 GB logical cores. TP16 doubles both (2 query heads, 4 experts, ~10 GB); so does
TP16 x CP2, the default, where ranks `r` and `r + 16` hold the same shard.

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

The scheduler, the CFG combine and the 133k x 4096 token embedding stay on the host.
That keeps a 1.1 GB table off every core, and means no uncompiled operation is ever
dispatched to the device under the Lite runtime.

The VAE decode runs on device by default (`model_config.vae_device: neuron`), as seven
staged float16 graphs with float32 reductions (`vae_device_dtype`), in a worker process on
a NeuronCore of its own (`vae_device_core`: 32 in the TP32 stage config, 16 in TP16) -- see
[Moving the VAE decode onto the device](#moving-the-vae-decode-onto-the-device). That core
must be outside the stage's `devices` range and otherwise unused. `vae_device: cpu`
decodes on the host in float32 (`vae_dtype`), which is the reference; the device path
falls back to it for any image wider or taller than 1024, past the NKI convolution's
spatial range, and for a request whose device decode fails.

### NKI kernels

| Kernel | Where | Fallback |
| --- | --- | --- |
| `attention_cte` (causal, d-major output) | prompt prefill attention | torch masked softmax |
| `output_projection_cte` | every attention output projection | `matmul` + bias |
| `moe_cte` (blockwise, `shard_on_block`) | routed experts | every local expert, densely |

Each kernel is gated on `can_run_kernel` and its own tiling limits, so CPU mode and
fake-tensor tracing take the torch path.

The MoE kernel is on by default (`model_config.moe_kernel: nki`, `moe_implementation:
shard_on_i_hybrid`, `moe_block_size: 512`; see [Which moe_cte loop](#which-moe_cte-loop));
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

`run.py --vae-device cpu` overrides the stage config's VAE placement for one run;
`--step-cache THRESHOLD` and `--tea-cache THRESHOLD` (0 disables either) override the
step caching the same way.

For think / recaption over HTTP, serve `hunyuan_image3_stage_ar.yaml`: its
`model_config.bot_task: think` applies to every request, because the diffusion chat
endpoint of vllm-omni 0.24 does not forward a request's `bot_task` (later releases pass it
through `extra_args`, which takes precedence):

```bash
bash examples/hunyuan_image3/serve.sh /models/HunyuanImage-3.0-Instruct 8091 \
  examples/hunyuan_image3/hunyuan_image3_stage_ar.yaml
```

A cold NEFF cache means the first request at a resolution compiles the VAE's seven stage
graphs in series inside that request, ~2 hours at 1024x1024. Compile them in parallel
beforehand instead -- one process per stage on cores 0-6, so stop the server first.
`--dtype` must match the stage config's `vae_device_dtype` (float16 by default):

```bash
python examples/hunyuan_image3/compile_vae_stages.py --height 1024 --width 1024
```

On the Lite runtime the worker pins the CCOM root (`NEURON_RT_ROOT_COMM_ID`) to
`localhost:61234 + <first core of the stage>`, the range vllm-neuron reserves for it, which
lies outside the kernel's ephemeral ports (32768-60999). Left to the backend, rank 0 picks
a free ephemeral port, releases it, and binds it only at the first collective; meanwhile
the same process opens ~20 listening sockets of its own from that range, and when one
lands on it the bind fails with `EADDRINUSE` (`Failed to schedule neff execution`, or
every other rank waiting on the root). That happened on 3 of 9 starts on one node before
the pin. Two servers on one node therefore need disjoint `devices`, which they need
anyway.

Per-request stage timings are written to
`$WORKLOAD_OUTPUT_RW/metrics/pipeline_perf_metrics.json` (`/tmp/metrics/...` by default):
`prefill_seconds`, `denoise_seconds`, `seconds_per_step`, `vae_decode_seconds` and
`e2e_forward_seconds`.

## Measured performance

**Current defaults** (2026-10-13): one `trn2.48xlarge`, TP16 x CP2, 1024x1024, 50 steps,
CFG 2.5, seed 42, the UNet split across the ranks and Cache-DiT at 0.12. Warm requests,
`run.py --profile --runs 3`:

| Request | Steps run / 50 | Denoise | VAE | Request | TP32 (2026-10-12) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DiT only (`hunyuan_image3_stage.yaml`), step caching off | 50 | 17.84-17.90 s (0.357 / step) | 1.87 s | 19.79-19.88 s | 29.89 s (0.558 / step) |
| **DiT only, default** | 25 | **10.66-10.68 s** | 1.87 s | **12.58-12.66 s** | 17.15 s |
| `bot_task: think` (`hunyuan_image3_stage_ar.yaml`), no CFG, no caching | 50 | 10.43-10.47 s (0.209 / step) | 1.87 s | 21.0-21.2 s (AR 8.7 s, 15.7 ms / token) | 27.48 s |

The last row is the setting of the GPU comparisons (vLLM-Omni, no CFG, no step cache):
the DiT stage there is 13.68 s on a g7e and 6.03 s on an H100, against 12.3 s here.

At TP32 the cached image was 32.1 dB PSNR to the uncached one for the DiT-only prompt and
33.8 dB for the think one, and the served and offline images were byte-identical (served
over `/v1/chat/completions`: 17.79-17.89 s DiT only, 28.57-28.60 s think); see
[Step caching](#step-caching) for the trade-off. The served path was not re-measured at
TP16 x CP2.

The tables below are the history of the earlier defaults. One `trn2.48xlarge`, TP32, 1024x1024, 50 denoising steps, BF16 backbone, CFG batch of 2
(`guidance_scale` 2.5), `moe_kernel: nki`. Measured 2026-10-11, from
`pipeline_perf_metrics.json` on warm generations. The default is the first column; the
other two are the same build with the VAE elsewhere:

The default (2026-10-12) is the first column: the `shard_on_i_hybrid` MoE loop (see
[Which moe_cte loop](#which-moe_cte-loop)) and the float16 VAE on its own core. The other
columns are the earlier `shard_on_block` MoE, with the VAE in each placement:

| Stage | **default** | `shard_on_block`, float16 VAE own core | `shard_on_block`, bfloat16 VAE shared core | `shard_on_block`, host VAE |
| --- | ---: | ---: | ---: | ---: |
| Prompt prefill | 0.02 s | 0.04 s | 0.04 s | 0.03 s |
| Denoise (50 steps) | **37.98 s (0.760 / step)** | 45.08 s (0.902 / step) | 45.01 s (0.900 / step) | 45.08 s (0.902 / step) |
| VAE decode | 1.83 s | 1.84 s | 2.49 s | 29.17 s (float32) |
| End to end (`e2e_forward_seconds`) | **39.89 s** | 47.03 s | 47.59 s | 74.34 s |
| `run.py --profile`, average | **39.94 s** (3 runs, 39.90-39.97) | 47.04 s (3 runs) | 47.62 s (2 runs) | 74.19 s (2 runs) |
| `diffusion_benchmark_serving.py`, 4 requests | not re-run | 47.55 s mean, P95 47.75 s | 48.22 s mean | -- |
| PSNR of the VAE decode to the host's, same latents | 66 dB | 66.41 dB | 58.92 dB | reference |

The float16 serving run: 4/4 successful, 190.20 s wall, median 47.54 s, P99 47.76 s,
`stage_0_gen_ms` mean 47020. The offline image and the one returned by
`/v1/chat/completions` for the same prompt and seed are byte-identical (float16
`sha256 a540ebeb...`, bfloat16 `73c6c5aa...`). Moving the decode saves 27.3 s, 37% of the
request; denoise is now 96% of it.

The measurements below predate the device VAE and are kept for the comparisons they
support. They used `moe_kernel: torch`, and the third warm generation:

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

### Which moe_cte loop

`examples/hunyuan_image3/bench_denoise_layer.py` times each piece of one decoder layer on
one core, at one rank's denoise shapes (CFG batch 2 x 4097 tokens, 1 query head, 2 of the
64 experts, routed top-8 of 64 so 12.5% of the pairs are live), each as its own graph:

| Piece | ms / layer | x 32 layers |
| --- | ---: | ---: |
| attention (QKV, RoPE, QK norm, two-bias attention, NKI o-proj) | 3.20 | 0.102 s |
| router | 0.36 | 0.012 s |
| shared expert slice | 0.64 | 0.021 s |
| norms + residuals | 1.74 | 0.056 s |
| routed experts, dense (both experts, every token) | 10.54 | 0.337 s |
| routed experts, `moe_cte` `shard_on_block`, block 1024 (former default) | 18.34 | 0.587 s |
| -- of which `build_blockwise_mapping` | 0.14 | |
| -- of which the kernel | 14.37 | |
| routed experts, `shard_on_i`, block 1024 | 15.81 | 0.506 s |
| routed experts, **`shard_on_i_hybrid`**, block 1024 / **512** / 256 | 5.92 / **5.19** / 5.60 | **0.166 s** |

The mapping builder was never the cost; the kernel loop was. The mapping is sized for the
worst case -- every token to every local expert, 19 blocks of 1024 here -- and
`shard_on_block` and `shard_on_i` walk all of them, live or not, which is more rows than
the dense path and explains why the blockwise kernel only ever reached parity.
`shard_on_i_hybrid` with `num_static_block=0` runs nkilib's hardware dynamic loop over
the live blocks only, which the mapping packs first. All three agree with float32 dense
math to bf16 level (5.7e-03 for both shard_on_i forms, 9.6e-03 for shard_on_block).
In the full model, per step on identical inputs, the hybrid path is as close to a
float32 CPU run as the dense path (see [Per-step check against
float32](#per-step-check-against-float32)).

In the model this takes a denoise step from 0.902 s to 0.760 s and a warm request from
47.04 s to 39.94 s; the step saving (0.142 s) is close to the per-layer one x 32 against
dense (0.171 s). Same prompt and seed, the image is 35.0 dB PSNR to the `shard_on_block`
one: a different accumulation order is amplified over 50 steps into different fine detail,
not degraded detail. The cold compile of the denoise graph grows to ~43 min.

The pieces sum to 0.35 s of compute per step. The rest of the 0.76 s was first put
down to the two all-reduces per layer; measuring them and profiling a step showed it was
the patch embed and final layer instead (next section).

### Where a step goes, and the UNet split

`examples/hunyuan_image3/bench_allreduce.py` times the TP32 all-reduce inside a compiled
graph, under serve.sh's runtime settings: 67 MB takes 1.07 ms (bus bandwidth 122 GB/s)
and 33.5 MB 0.86 ms, so the 64 per step are ~68 ms. Two independent branches in one
graph -- all-reduces and matmuls -- do not overlap: together they take 4-8 ms longer than
apart, with or without `--enable-ccop-compute-overlap`, so interleaving the CFG branches
to hide communication would only add the cost of halving each all-reduce.

A device profile of a warm step (vllm-neuron's `NeuronProfiler`, ranks 0/1/16/31) put the
760 ms at:

| Part | Before | With `unet_parallel` |
| --- | ---: | ---: |
| Patch embed (`UNetDown`) and layer 0's attention, to the first all-reduce | 66 ms | 9.2 ms |
| 32 decoder layers (collectives 109 ms, of which ~41 ms waiting on peers) | 516 ms | 544 ms |
| Final layer (`UNetUp`) | 179 ms | 4.4 ms |
| Step | 760 ms | 557 ms |

The patch embed and final layer are ResBlocks over the 64x64 latent grid with up to 4096
channels, ~4 TFLOP per CFG step, and every rank ran them whole; the final layer also
spent ~167 ms in small DMAs turning the 67 MB token tensor back into an image layout.
`unet_parallel.py` splits them by GroupNorm group -- all 32 groups, one per rank at
TP32, so the statistics stay local -- with column-parallel convolutions over an
all-gathered input, the timestep projection producing only the rank's scale/shift, and
transposes only of the rank's own channels. A step goes from 0.7585 s to 0.5578 s and a
warm request from 39.94 s to 29.89 s. On identical inputs, step by step, it is 5.25e-02
from float32 against 5.64e-02 for the replicated path, and the images are 35.2 dB apart
-- the usual drift of a reordered accumulation over 50 steps. A fused GroupNorm + SiLU
kernel, as vllm-omni has for CUDA, would save less than 0.5 ms a step after the split.

### Context parallelism

With the UNet split, the 32 decoder layers are the step, and a TP32 profile of a no-CFG
step put 69 ms of its 329 ms in the 64 all-reduces and another ~40 ms in work every TP
rank repeats on every token (RMSNorm, the float32 router). `context_parallel: 2` runs the
backbone at TP16 over ranks 0-15 and 16-31 -- one row of the trn2 chip torus each -- and
gives each half the tokens: CP pairs `(r, r + 16)` are neighbouring chips at the same
core offset. In each layer the pair all-gathers its image K/V (one KV head per rank, 1 MB);
the final layer gathers the tokens back, and the UNet stays split over all 32 ranks.

`bench_allreduce.py --only tpcp` measured the collectives first: a 16.8 MB all-reduce
within a TP16 group, both groups at once, takes 0.381 ms (64 a step: 24 ms, against 55 ms
for 33.5 MB at TP32), and the K/V all-gather 0.071 ms (2.3 ms a step). A step went from
0.338 s to 0.264 s without CFG and from 0.558 s to 0.453 s with it. On identical inputs
the CFG prediction is 5.26e-02 from float32 at steps 0/25/49 (TP32: 5.25e-02) and
3.60e-02 from the hybrid-MoE trajectory over 50 steps (TP32: 3.49e-02); the free-running
image is 34.5 dB from the teacher-forced one, the usual reordering drift.

The AR decode is bound by reading weights, and at TP16 each rank would read twice as many
(23.8 ms a token against 15.5 ms at TP32). So the two ranks holding a shard split it for
decode -- half the query heads, local experts, shared-MLP columns and LM-head rows each --
and all-reduce over all 32 ranks: 15.7 ms a token. Greedy decoding is still exact, but the
reduction order differs from TP32, so the think text (and with it the image) can differ
from a TP32 run's.

The token split pads the 4097 step tokens (timestep token + 4096 image tokens) to 2 x 2049;
the padding key is masked and only CP rank 0's first query is the timestep token. The CP
gathers run one per batch row: one gather of the whole CFG batch followed by a
`[cp, batch] -> [batch, cp]` transpose made neuronx-cc fail on the 32-layer CFG graph
(`NCC_IBTN020 convertAccessPattern: step out of int16 range`) while 2 layers compiled.
`context_parallel` must divide the world, and the TP size must still divide the 32 query
heads, 64 experts and the vocabulary; TP8 x CP4 would not fit (~20 GB of weights a core).

With CFG, `cfg_parallel` (default on) uses the two TP16 blocks differently: block 0 runs
the cond branch and block 1 the uncond one, each at batch 1 over all 4097 tokens, with no
K/V exchange and each block prefilling only its own branch. Both branches share the
latents and timestep, so the patch embed stays split over all 32 ranks at batch 1; the
final layer gathers the two branches' hidden states into the CFG batch and runs over all
32 ranks, so every rank returns both predictions. Which mode a step graph is in follows
from its RoPE table's length (a share of the tokens, or all of them). Measured on the
same code, Mars prompt, CFG 2.5: 0.452 s a step with the token split, 0.442 s with
`cfg_parallel`; with cache-dit 0.12, 14.14 s and 13.94-14.01 s a request, caching the same
25 steps. Teacher-forced, the CFG prediction is 5.40e-02 from float32 at steps 0/25/49 and
3.58e-02 from the hybrid-MoE trajectory over 50 steps. (Timing one TP16 block alone on 16
cores, 0.496 s a step, is not a stand-in for this: there the UNet split is over 16 ranks.)

### Denoise attention kernel

`denoise_attention: nki` (the default) runs the denoise step's image queries on nkilib's
`attention_cte` instead of a torch softmax the compiler fuses, which materialised the
float32 `[heads, queries, keys]` score matrix and spilled it (59 ms and 7.1 GB a step at
TP32 without CFG). The kernel's prefix caching fits the step: the right-padded prompt K/V
is the prefix, `prior_used_len` -- a runtime value, read off the image key bias -- masks
its padding, and the step's own K/V is the non-causal active part. That needs every image
token to see the whole prompt and the whole step, which the pipeline checks against the
generation mask (and refuses the kernel otherwise). The timestep token sees only the
prompt and itself, so its one row stays in torch with its own bias.

`check_nki_kernels.py --tp-size 16 --only denoise_attention` compares it with the torch
path on one core: 3.5e-03 relative at 1 x 2049 queries x 4097 keys, 4.1e-03 with 8 of 512
prompt slots live, 4.2e-03 at 1 x 4097 x 4097 with a 2048 bucket. In the model a step goes
from 0.442 s to 0.357 s with CFG (CFG parallel) and from 0.264 s to 0.209 s without;
teacher-forced, the CFG prediction is 5.27e-02 from float32 (torch attention: 5.40e-02).
`denoise_attention: torch` keeps the fused-softmax path.

### Step caching

`step_cache` is Cache-DiT's DBCache (`Fn_compute_blocks` 1, `Bn_compute_blocks` 0) on the
host loop, with vllm-omni's `CacheDiTConfig` names and rules: every step runs the patch
embed and layer 0; if layer 0's residual moved less than `residual_diff_threshold` since
the last full step (`mean|R_prev - R| / mean|R_prev|`), the other layers are skipped and
their last residual added back -- never in the first 4 steps, never more than 3 in a row.
The step is three graphs (front, rest, hit) and the host picks one per step; the ranks
compute identical diffs and agree over gloo before every step. A computed step costs
0.565 s, a cached one 42 ms.

`tea_cache` is the TeaCache vllm-omni ships for this model (in its HunyuanImage3
pipeline's denoise loop): the rescaled relative change of `time_embed(t)` accumulates
until `rel_l1_thresh`, and steps below it reuse the last prediction without touching the
device. Coefficients and the 0.2 default come from vllm-omni's `TeaCacheConfig`. The
decision depends only on the timesteps, so it is planned once per schedule.

Mars prompt, seed 42, CFG 2.5; PSNR to the uncached image:

| Setting | Steps run / 50 | Denoise | Request | PSNR |
| --- | ---: | ---: | ---: | ---: |
| off | 50 | 27.89 s | 29.89 s | -- |
| `step_cache` 0.08 | 33 | 19.36 s | 21.35 s | 34.05 dB |
| **`step_cache` 0.12 (default)** | 25 | **15.17 s** | **17.15 s** | **32.14 dB** |
| `step_cache` 0.24 (Cache-DiT's default) | 17 | 10.95 s | 12.93 s | 27.28 dB |
| `tea_cache` 0.1 | 17 | 9.58 s | 11.54 s | 25.87 dB |
| `tea_cache` 0.2 (vllm-omni's default) | 10 | 5.68 s | 7.67 s | 18.93 dB |

At 0.08 and 0.12 the image is visibly the same; at Cache-DiT's 0.24 it keeps its
composition but loses fine structure. TeaCache reuses whole predictions from the
timestep alone, so at the same speed it moves the image further: at 0.2 the dome's shape,
the base and the light change. Its coefficients are calibrated on GPU, not on trn2.

### Against the NxDI port of the same model

An earlier port of HunyuanImage-3.0 to the same `trn2.48xlarge` ran on NxDI with the cores
split AR16 + DiT16 + VAE1, at the same 1024x1024 / 50 steps / CFG 2.5:

| | NxDI port | This port (Lite) | |
| --- | ---: | ---: | --- |
| Denoise | 71.846 s (**1.437 / step**) | 44.754 s (**0.895 / step**) | **1.61x** |
| VAE decode (host) | 27.461 s | 26-27.8 s | same |
| End to end | 122.625 s | 72.10 s | 1.70x |

Only the **denoise** row is a strict comparison, and it is the 1.61x. The end-to-end row
is not like for like in two ways: the NxDI port spends 20.805 s generating 261 AR tokens
on device for recaptioning, which this pipeline's causal prompt prefill (0.053 s) does not
do; and its numbers are single functional requests, where these are three-run warm
averages. So the honest claim is 1.61x on the phase both stacks compute the same way.

The VAE row being identical is not a coincidence — the NxDI port tried to put the decoder
on device too and hit the same wall, 11,377,628 instructions against the same 10,000,000
`NeuronHloVerifier` ceiling (`evidence/nxdi-vae1024-compiler-limit.log` in the porting
notes). Two independent stacks, twelve days apart, rejected by the same verifier at
counts that differ only by graph construction. That is what makes this a property of the
decoder rather than of how either stack traces it, and the per-stage split below is the
approach the NxDI port did not try.

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

That left the host-side VAE decode as the largest remaining term, at 36% of a warm
request -- which is what the next section removes.

### Moving the VAE decode onto the device

The decode was the largest remaining term in a warm request: 29.6 s of float32 on the
host, 36% of end to end, with no host-side slack left (96 threads, no `OMP_NUM_THREADS`
cap). It now runs on device in **2.59 s, 11.4x faster**, measured by
`examples/hunyuan_image3/check_vae_decode.py`, which loads the real VAE weights, decodes
the pipeline's actual latent shape on the host as a reference, then runs the same decode
on Neuron and reports per-stage time, speedup and error.

| Stage | Output | Host float32 | Device | Speedup | Error | Image frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `in+mid` | 1024 ch, 64x64 | 0.15 s | 0.01 s | 22.4x | 5.57e-03 | 5.57e-03 |
| `up0` | 1024 ch, 128x128 | 0.36 s | 0.05 s | 7.8x | 7.55e-03 | 6.59e-03 |
| `up1` | 512 ch, 256x256 | 2.11 s | 0.21 s | 9.9x | 7.82e-03 | 6.53e-03 |
| `up2` | 256 ch, 512x512 | 5.41 s | 0.46 s | 11.9x | 7.66e-03 | 8.14e-03 |
| `up3` | 128 ch, 1024x1024 | 8.80 s | 0.75 s | 11.8x | 8.42e-03 | 8.59e-03 |
| `up4` | 128 ch, 1024x1024 | 10.83 s | 1.03 s | 10.5x | 9.48e-03 | 7.18e-03 |
| `out` | 3 ch, 1024x1024 | 1.09 s | 0.09 s | 12.4x | 8.47e-03 | 1.64e-02 |
| **Decode** | | **29.6 s** | **2.59 s** | **11.4x** | | **1.64e-02** |

Four things had to be true at once, and each was found by an experiment that ruled the
previous answer out.

**The graph was too big, and cutting the image up made it worse.** The whole decode is
10,746,838 instructions against `NeuronHloVerifier`'s 10,000,000 ceiling
(`NCC_EVRF007`) -- and 11,852,476 at 512x512, 17,488,800 for a single 384x384 spatial
tile. The count tracks the decoder's op count and the compiler's per-op expansion at small
spatial dims, not the data volume, so `NCC_IXTP002`'s advice ("Tiling could potentially do
a better job") is backwards here; spatial tiling also loses on the host, 56.98 s against
29.22 s, from decoding 16 overlapping tiles. `--internal-max-instruction-limit` does not
help either: it reaches `walrus_driver`, but the rejection happens earlier. An earlier
NxDI port of this model hit the same wall at 11,377,628 instructions, which is what makes
this a property of the decoder rather than of how either stack traces it.

**One convolution was most of the graph.** Routing the 3x3x3 convolutions through
`nkilib.experimental.conv.conv3d` -- the same kernel and the same `nki_op`/`wrap_nki`
dispatch the Wan2.2 VAE in this repo uses -- made the compiler name the culprit:
`[NCC_EXTP003] Instructions generated by compiler 14155776 exceeds the typical limit of
300000` at `Decoder/UpsampleDCAE[upsample]/Conv3d[conv]_convolution`. `UpsampleDCAE`
upsamples by convolving to `C * r**3` channels and rearranging, so its two convolutions
have `C_out` 8192 and 4096, over the kernel's 2048 ceiling. Since output channels are
independent, computing them in groups and concatenating is exact, so the wide banks are
split across 4 and 2 launches. That one change took the stage that could not compile at
all down to a 74 s compile.

**Depth, not area, is the axis to split on.** Each stage -- `conv_in`+mid, each upsample
level, the output head -- compiles as its own graph with a fraction of the ops at full
spatial size, with intermediates staying on device between stages. Compile times fell with
it: 151 s to 16 s for `in+mid`, 40 min to 13 min for `up1`.

**bfloat16 broke the image, and the cause was the reduction, not the mantissa.** A first
bfloat16 run was 11.6x faster and numerically useless: 1.0265 relative error on the
decoded frame. The kernel was not at fault -- compared against `F.conv3d` on device at
the real decoder shapes it is 6.3e-07 to 9.2e-07 in float32 and 3.2e-03 in bfloat16,
including the split case, with `out/ref` 1.000. And 3.2e-03 per convolution cannot
compound into a 98% error over 44 layers; an unstable step has to exist. GroupNorm is it,
twice over: its last level reduces 4 channels x 4 frames x 1024^2 ~ 16.8M elements, and a
low-precision running sum stops absorbing terms once it exceeds them by ~256x, while
computing the variance as `E[x^2] - E[x]^2` cancels catastrophically when the mean
dominates and can reach `rsqrt` at zero. Mantissa width sets how good the result can be,
not whether it breaks. Accumulating the statistics in float32 -- two passes, via
`mean(dtype=torch.float32)` so no float32 copy of the activation is materialised, with
`eps` added in float32 -- brings the image to **1.64e-02**, which is the per-operator
bfloat16 level and flat across all seven stages. This is the half of
`torch.autocast(bfloat16)` the reference relies on and the backbone already reproduced in
`_group_norm_f32`; the VAE was missing it. `torch.autocast("neuron")` cannot supply it:
that device hook is a numeric no-op shim. The mid block's attention softmax is upcast for
the same reason.

So float16 was never needed. It would have brought 10 mantissa bits against bfloat16's 7,
but the failure was never precision, and float16's narrow range is a risk this decoder
does not have to take.

**float32 does not fit, and the reason is measurable.** With everything above in float32
the decode holds 13.1 GiB on a core that has 24 GiB, because the decoder's float32
weights are 3.25 GiB and the routed 3x3x3 convolutions are 3.23 GiB of them -- so each was
resident twice, as `module.weight` and as the packed NKI filters holding the same values.
`up2` still ran and was exact (6.61e-07) but took **20.51 s**, four times slower than the
host, and `up3` then failed with `nrt_tensor_allocate status=4`. Releasing the duplicate
weights after packing, building the decoder alone (`only_decoder`, the encoder is 1.45 GiB
and decode never calls it), and freeing each finished stage's graph brings bfloat16
residency to **5.0 GiB**, measured, with ~19 GiB spare.

**In the pipeline.** `vae_device: neuron` builds this decoder on the output rank's core
on the first request (`vae_device.StagedDeviceDecoder`, 5.0 s from the host VAE's weights,
which are already loaded) and runs the seven stages back to back. The check script and the
pipeline share `vae_device.py`, so they trace the same graphs. A warm decode is **2.49 s**
and a warm request **47.6 s**, against 29.17 s and 74.2 s on the host path in the same
build; the ~48 s projection held. Same prompt and seed, the device image is **58.92 dB**
PSNR against the host float32 decode: at most 3/255 off per channel, 0.083/255 on average,
77.74% of pixels identical -- in line with the 58.71 dB the standalone check measured on a
real image, and a tenth of the VAE's own 47.04 dB round-trip error.

**Memory on the output rank's core**, from `neuron-monitor`: every core peaks at 7.73-7.76
GiB with the host decode, so the output rank had 16.2 GiB spare on its 24 GiB core
(96 GiB per device over 4 logical cores at LNC2). With the decoder resident it peaks at
**19.93 GiB** -- model code 5.00, shared scratchpad 5.00, tensors 9.86 -- leaving 4.1 GiB.
The other 31 cores are unchanged. Most of the growth is the stage NEFFs' instruction streams
and a larger scratchpad, not the 5.0 GiB of decoder tensors the standalone check measured,
so measure with `neuron-monitor` rather than adding up weights.

**float16, on a core of its own.** float16 carries three more mantissa bits than
bfloat16 at the same width, and here that is worth 7x: on the real-image check the frame
error falls from 1.31e-02 to 1.47e-03, PSNR to the host decode rises from 58.71 dB to
66.50 dB, pixels identical to the host from 76.85% to 95.71%, at most 1/255 off. In the
pipeline it is 66.41 dB against 58.92. The time is no worse -- 1.6-1.8 s against 2.5 s
for bfloat16, a difference in how the two dtypes compiled rather than anything the dtype
itself buys.

Its range is the risk. On the host in float32, the decoder's activations peak at 26 / 18 /
55 / 1270 / **9559 / 9721** / 6.6 across the seven stages (the same within 2% on a second,
very different image) -- 6.7x inside float16's 65504. But a GroupNorm squares its centred
input, and anything past 256 squares past 65504. On device the current compiler happened
to keep that square in float32 (a GroupNorm fed values up to 3e4 stayed finite), but
eager float16 on the CPU overflows exactly as the arithmetic says it should, and nothing
guaranteed the compiler would keep doing that. So the float16 GroupNorm scales each group
by a power of two that brings `|x - mean|` to at most 1 before centring, and normalises
the scaled values directly; the result is scale-invariant once `eps` is scaled with it.
A power of two is exact, so this adds no rounding -- on device it matches the unscaled
version to the digit (6.06e-04 / 6.33e-04 / 6.28e-04 against float32 at peaks of 200 /
9700 / 30000) -- and nothing larger than 1 is ever squared.

It does not fit beside the backbone. The runtime keeps far more of the decoder resident
than its 1.6 GiB of weights: the seven stage NEFFs' code and DMA rings (4.5 GiB), a shared
scratchpad sized by the largest stage (5.75 GiB in float16, where `up3` compiles to 0.75
GiB more than in bfloat16), and an output buffer per stage that stays allocated between
decodes (2.8 GiB). That last part is below PyTorch: no Python object references it,
`empty_cache()` does not return it, and a second decode reuses it rather than growing. The
first request loads the stages one at a time, so it fits; the second has all seven
resident, and float16 then failed to allocate `up3`'s 1 GiB output on the output rank's
core (`nrt_tensor_allocate`, 21.3 GiB in use). bfloat16 had fitted with less to spare than
the 4.1 GiB measured above suggested -- `neuron-monitor` at 2 s intervals misses the
transient peak.

So the decoder now runs in a worker process (`vae_worker.py`) on a core outside the stage's
`devices` range, which a TP32 deployment leaves idle. The output rank starts it on the
first request (36 s to load the decoder and attach the core), sends each request's
latents over a pipe and gets the float32 frame back; the transfer is inside the 1.84 s. It
builds the same `StagedDeviceDecoder` from the same file, so it hits the same NEFFs. The
VAE core peaks at **14.72 GiB of 24**, and every backbone core, the output rank's
included, is back to 7.76 GiB. The worker exits when the pipe closes, so it cannot outlive
the server.

**The NEFF cache is keyed on source lines, not just on the graph.** torch-mlir attaches
each op's location -- the absolute path and line of every traced frame -- to the MLIR, and
the cache key is a hash of that MLIR. Moving the stage functions out of the check script
into `vae_device.py` was therefore a cold compile, even though the graph had not changed.
Copying the same code to another directory was a cold compile as well. So a saved
`neff-cache` is only valid for the same files at the same path; with this repo installed
at `/workspace/plugin` (the bootstrap layout), editing a line above a traced function
costs that function's compile. The prefill graph, whose code is in an unchanged file,
hit the cache across the same edit that recompiled the denoise step. The key does
**not** depend on the caller, the compile order or `NEURON_CC_FLAGS` (the stage's
`compiler_args` replace it). A stage compiled by `compile_vae_stages.py` from a zero
tensor hits the cache from a different script that compiled an unrelated graph first
(0.9 s against 15 s cold). That is what makes compiling the stages in parallel valid: the
seven took 47 min wall at 1024x1024 (1 s for the cached `in+mid`, then
115 / 891 / 1559 / 2741 / 2263 / 116 s), against ~2 hours in series.


## Think / recaption (AR + DiT)

Upstream runs the text generation as a separate LLM stage on separate devices (AR, then
DiT, joined by a KV connector). Here it runs on the DiT's own 32 cores and weights, in
`hunyuan_image3_ar.py`, before the denoise loop: the existing prompt prefill over the
prompt minus its last token, then a one-token decode graph replayed per token. The decode
step's generated-K/V cache is a fixed `[1, ar_max_new_tokens, kv_heads, head_dim]` buffer
per layer, updated as `cache * (1 - onehot) + new * onehot` -- exact for a one-hot, and
not an in-graph slice write, the construct that broke `neuronx-cc`'s MaskPropagation pass
in this backbone's MoE. The position, RoPE row and visible key range arrive as inputs, so
the graph never changes shape. The decode MoE evaluates the rank's two local experts
densely (the blockwise kernel would pad one token to a 1024-row block), and the LM head is
vocab-parallel: each rank holds 4160 rows and one all-reduce of the zero-padded shards
gives every rank the full logits, so all ranks take the same greedy token.

The protocol follows upstream's AR stage: `prompt_utils.build_prompt_tokens` builds the
prompt (`en_unified` system prompt, `User:`/`Assistant:` turns, trigger tag), decoding is
greedy as in `deploy/hunyuan_image3_ar.yaml`, the stop tokens come from
`resolve_stop_token_ids` for an explicit size, `</think>` forces `<recaption>` under
`think_recaption`, and the text is truncated at the closing tag and handed to the DiT as
`ar_generated_text` with `bot_task` back to `auto`, as `ar2diffusion` does.

```bash
python examples/hunyuan_image3/run.py --bot-task think \
  --stage-config examples/hunyuan_image3/hunyuan_image3_stage_ar.yaml
```

`hunyuan_image3_stage_ar.yaml` is the default stage config with `prefill_len: 2560`
(the AR prompt is 1257 tokens, and the DiT prompt then carries the generated text too) and
`ar_max_new_tokens: 1024`.

With the current defaults (UNet split, Cache-DiT 0.12) the same request is **28.6 s** over
`/v1/chat/completions` (AR 8.4 s, denoise 17.5 s, VAE 1.9 s); see
[Measured performance](#measured-performance). The 2026-10-11 measurement below predates
both and the `shard_on_i_hybrid` MoE loop.

Measured 2026-10-11, 1024x1024, 50 steps, CFG 2.5, seed 42, the prompt "A cozy wooden
cabin in a snowy pine forest at dusk, warm yellow light glowing from the windows, smoke
rising from the chimney", `bot_task: think`, warm (`run.py --profile --runs 2`):

| Stage | trn2 (this pipeline) | H100 x8, upstream 4+4 (2026-09-30) |
| --- | ---: | ---: |
| AR prompt / generated tokens | 1257 / 547 | 1257 / 548 |
| AR (prefill + decode) | **8.80 s** (15.5 ms / token) | 35.85 s (65.4 ms / token) |
| DiT denoise, 50 steps | 49.68 s (0.994 / step, CFG 2.5) | 6.03 s with the VAE (`guidance_scale: 0`, no CFG) |
| VAE decode | 1.87 s | (in the line above) |
| End to end | **60.56 s** (60.55 / 60.57) | 41.98 s |

The AR is ~4x faster than H100 here: batch-1 decode is bandwidth-bound and 32 cores split
the weights 32 ways. The DiT is what remains slow -- 0.994 s/step against 0.90 for the
DiT-only prompt, because the prompt now carries ~550 more tokens -- and the H100 row ran
without CFG, half the per-step work. The image follows the generated text (trees framing
the scene, weathered logs, warm light on the snow, chimney smoke). The first AR request
compiles the decode graph and the 2560-token prefill and denoise graphs: 1314 s cold.

`test/unit/test_hunyuan_image3_ar_decode.py` checks the decode step against a full causal
forward of the reference math on CPU, token by token: 4.9e-07 relative, same argmax.

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

`test/unit/test_hunyuan_image3_vae_device.py` checks the staged decode on a small random
VAE of the same architecture, on CPU: the seven stages run back to back equal
`vae.decode` exactly (frame selection included), and the float32-statistics GroupNorm and
attention match the stock modules to 5.3e-07 in float32. In float16 at a peak of 9700 --
where the test first confirms the naive square overflows -- the GroupNorm stays finite
and within 5.6e-04 of float32, and a constant group gives its bias rather than inf * 0.

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

### Per-step check against float32

Comparing final images says little about a kernel change: 50 steps amplify any
difference in accumulation order into different fine detail. Instead, the pipeline can
record every denoise step and replay recorded inputs (`_StepTrace` in
`pipeline_hunyuan_image3.py`). With `HY3_TRACE_DIR` it saves each step's input latents,
the raw CFG-batch prediction and the combined prediction. With `HY3_TRACE_FORCE_DIR` it
feeds a recorded trace's input latents into every step (teacher forcing), so two
implementations are compared on identical inputs at every step. With `HY3_TRACE_STEPS`
it runs only the listed steps. `examples/hunyuan_image3/check_denoise_reference.py` runs
the same pipeline class on the host at TP1 in float32, with every kernel on its torch
fallback. Each step takes ~20 min there; the model loads in ~6 min and needs ~320 GB of
host memory. `examples/hunyuan_image3/compare_step_traces.py` diffs traces:

```bash
# reference trajectory on device, then the others forced onto its latents
HY3_TRACE_DIR=/workspace/out/trace/hybrid python examples/hunyuan_image3/run.py ...
HY3_TRACE_FORCE_DIR=/workspace/out/trace/hybrid HY3_TRACE_DIR=/workspace/out/trace/dense \
  python examples/hunyuan_image3/run.py --moe-kernel torch ...
HY3_TRACE_FORCE_DIR=/workspace/out/trace/hybrid HY3_TRACE_STEPS=0,25,49 \
  HY3_TRACE_DIR=/workspace/out/trace/cpu_fp32 \
  python examples/hunyuan_image3/check_denoise_reference.py --seed 42
python examples/hunyuan_image3/compare_step_traces.py \
  --reference cpu_fp32=/workspace/out/trace/cpu_fp32 hybrid=... block=... dense=...
```

The test case is the Mars-observatory prompt above at seed 42, 1024x1024. It compares
the three routed-expert paths on device (bf16, TP32): `shard_on_i_hybrid` (the default),
`shard_on_block`, and the dense torch path (`moe_kernel: torch`). Each number is the
relative L2 error of the combined CFG prediction, which is what the scheduler consumes.
The conditional and unconditional branches are given in brackets:

| vs float32 CPU | step 0 | step 25 | step 49 | mean | cosine min |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense (torch) | 4.08e-02 | 3.80e-02 | 8.59e-02 | 5.49e-02 (2.20e-02 / 2.98e-02) | 0.99630 |
| `shard_on_block` | 4.06e-02 | 3.74e-02 | 8.73e-02 | 5.51e-02 (2.23e-02 / 2.91e-02) | 0.99619 |
| **`shard_on_i_hybrid`** | 4.77e-02 | 3.82e-02 | 8.32e-02 | 5.64e-02 (2.34e-02 / 2.98e-02) | 0.99653 |

| vs dense bf16, all 50 steps | mean | max | cosine min |
| --- | ---: | ---: | ---: |
| `shard_on_block` | 3.47e-02 (1.22e-02 / 1.23e-02) | 8.08e-02 | 0.99673 |
| `shard_on_i_hybrid` | 3.46e-02 (1.23e-02 / 1.22e-02) | 8.10e-02 | 0.99671 |

All three paths are the same distance from float32, within ±3% of one another. No path
is consistently best: hybrid is worst at step 0 and best at step 49. Their distance from
each other (3.5e-02) is about the same as their distance from float32, so that much error
is simply what bf16 gives. The MoE kernel adds nothing measurable on top of it. The
combined error is ~2x the branch errors, because the CFG combine
`uncond + 2.5 (cond - uncond)` amplifies the difference between two close branches. It
peaks at the last step (t ≈ 9) for every path, float32 aside.

## Known limits

- One resolution per compiled graph. Changing height/width, `prefill_len`, the step
  layout or the TP degree triggers a cold NEFF build; point
  `TORCH_NEURONX_NEFF_CACHE_DIR` at storage that outlives the container.
- A prompt longer than `prefill_len` is an error, not a silent truncation. Raise the
  bucket (and pay one recompile).
- CFG parallelism needs the two TP16 blocks of `context_parallel: 2`; at TP32 or TP16
  both CFG branches run in the same batch. `guidance_scale <= 1.0` runs a single branch.
- Step caching is on by default and changes the image (32 dB PSNR to the uncached one at
  0.12). Set `step_cache: false` for every step to run.
- Over HTTP a request cannot choose `bot_task` with vllm-omni 0.24; the stage config's
  `model_config.bot_task` applies to all requests.
- The TP16 stage config carries the same defaults (`unet_parallel`, `step_cache`) but was
  not re-measured with them on device. It stays at `context_parallel: 1`: CP over 16
  cores would mean TP8, which does not fit.
- No conditioning-image (IT2I / image edit) path, no FP8.
- The AR is greedy only, for an explicit image size (no `auto` aspect-ratio prediction),
  one request at a time, and serial with the DiT on the same cores rather than pipelined.
  Its numerics are checked against the reference math, not against an H100 token stream.
- The device VAE needs one more NeuronCore than the stage's `devices` (`vae_device_core`).
  Without one, leave `vae_device_core` unset and set `vae_device_dtype: bfloat16`: it then
  shares the output rank's core, which bfloat16 fits beside and float16 does not.
- The VAE decode runs on one core, not across ranks (1.8 s at 1024x1024). Images over
  1024 on either side decode on the host (29 s), because the NKI convolution's spatial
  range ends there and the device decoder has released the torch weights those sites
  would need.
- Each resolution and dtype the device VAE decodes is seven more NEFFs: ~30-47 min to
  precompile at 1024x1024 with `compile_vae_stages.py`, ~2 hours if the first request
  compiles them. One resolution in float16 holds 14.7 GiB of the VAE core's 24.
- The NEFF cache key includes source paths and line numbers (see
  [Moving the VAE decode onto the device](#moving-the-vae-decode-onto-the-device)), so a
  cache saved from one checkout is only reused by the same files at the same path.
- The routed experts take 5.2 ms of a layer's ~16 ms with `shard_on_i_hybrid`, half the
  dense path; the rest of a layer is attention, norms, the shared expert and the two
  all-reduces (see [Which moe_cte loop](#which-moe_cte-loop)). The step's matmul
  utilization is low (6% in the profile) and the tensor engine runs power-throttled about
  half the time, which is where further per-step gains are.
- An earlier report that the dense MLP kernel "wedged a NeuronCore" was wrong. It raises
  a clean compile-time validation error; the node losses that coincided with it have a
  separate explanation (a standing health-agent flag plus node auto-recovery).

## Related information

- [Onboarding a model](../model-dev/onboarding-models.md) — the process this
  implementation follows.
- [Setup guide](../getting-started/setup-guide.md) — environment and container.
- [Wan2.2-T2V-A14B model card](wan22-t2v-14b.md) — the reference blueprint.
