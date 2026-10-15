# HunyuanImage-3.0-Instruct Model Card

<!-- meta: description: Model card for HunyuanImage-3.0-Instruct on AWS Trainium2 with
the vLLM Omni Neuron plugin — DiT-only text-to-image at 1024x1024 with classifier-free
guidance on one trn2.48xlarge (TP16 x CFG2), the NKI kernels and on-device VAE decode,
the three-tier accuracy method, results, known limitations and the roadmap. -->
<!-- meta: keywords: vLLM Omni, Neuron, Trainium, trn2, HunyuanImage-3.0,
HunyuanImage3ForCausalMM, model card, text-to-image, DiT, MoE, tensor parallelism,
CFG parallelism, NKI, attention_cte, moe_cte, VAE, torch.compile -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-15 -->

## Introduction

[HunyuanImage-3.0-Instruct](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) is
Tencent's 80B-parameter mixture-of-experts multimodal model (64 routed experts, top-8;
~13B active per token). One decoder-only backbone both generates text and denoises image
latents: the prompt is prefilled causally, then the image tokens are denoised by flow
matching with full attention among themselves.

This build supports the model's **DiT-only text-to-image** path for inference serving with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on one
AWS Trainium2 `trn2.48xlarge`: a text prompt in, one 1024x1024 image out, with
classifier-free guidance (the checkpoint defaults: guidance 2.5, 50 steps). The prompt goes
straight to the diffusion stage with `bot_task: image` — no think / recaption text and no
AR stage — the scope of upstream vLLM-Omni's `vllm_omni/deploy/hunyuan_image3_dit.yaml`.
Image editing and think / recaption are on the [roadmap](#not-in-this-build--roadmap).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| HunyuanImage-3.0-Instruct | [tencent/HunyuanImage-3.0-Instruct](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) (validated at revision `2ec2c78bee7d4b94157341fba86c4c2c7b1858b2`) | Trn2 | BF16 |

Registered architecture key: `HunyuanImage3ForCausalMM`. The Neuron pipeline
(`NeuronHunyuanImage3Pipeline`) subclasses upstream's `HunyuanImage3Pipeline` for that key
and keeps its request handling (chat template, 2D RoPE, generation mask), replacing
everything from `forward()` down.

| Property | Value |
| --- | --- |
| Decoder layers | 32 |
| Hidden size | 4096 |
| Attention | 32 query heads / 8 KV heads, head dim 128, QK RMSNorm |
| Position encoding | 2D RoPE (theta 10000), applied before the QK norm |
| Routed experts | 64, top-8, renormalised, intermediate 3072 |
| Shared expert | 1, intermediate 3072 |
| Latents | 32 channels, VAE spatial factor 16 (1024x1024 -> 64x64 -> 4096 tokens) |
| Scheduler | FlowMatchEulerDiscrete, shift 3.0 |
| Checkpoint dtype | BF16 |

## Features

Per-model feature availability for HunyuanImage-3.0-Instruct in this build. See the
[README](../../README.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image (`bot_task: image`) | ✅ |
| | 1024x1024 | ✅ |
| | Other resolutions | ❌ |
| | Image editing / IT2I | ❌ (roadmap) |
| | Think / recaption on the backbone | ❌ (roadmap) |
| **Quantization** | BF16 | ✅ |
| | FP8 | ❌ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | CFG Parallelism (`cfg_parallel_size: 2`) | ✅ |
| | Context Parallelism (`ring_degree`) | ❌ (roadmap) |
| | Megatron Sequence Parallelism (`tp_sequence_parallel`) | ❌ (roadmap) |
| | VAE decode row split (`vae_parallel: 2`) | ✅ |
| | [VAE Patch Parallelism](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/vae_parallel/) (`vae_patch_parallel_size`) | ❌ (roadmap) |
| **Performance** | Classifier-Free Guidance | ✅ |
| | VAE decode on device | ✅ |
| | Step caching (Cache-DiT, TeaCache) | ❌ (roadmap) |
| | Continuous request batching | Limited |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated in this build; validation status in [Results](#results)
- Limited: concurrent requests are accepted but run serially (`max_batch_size: 1`)
- ❌ Not supported in this build; requests and `model_config` keys for these features
  raise an error (see [Known limitations](#known-limitations))

### Recommended configuration

The only stage config is
[`hunyuan_image3_dit.yaml`](../../examples/hunyuan_image3/hunyuan_image3_dit.yaml): the 32
logical NeuronCores of one `trn2.48xlarge` (`NEURON_LOGICAL_NC_CONFIG=2`,
`devices: "0-31"`) as `tensor_parallel_size: 16` x `cfg_parallel_size: 2`, with
`ring_degree: 1` and `vae_patch_parallel_size: 1`.

- **CFG parallelism.** The two TP16 replicas are the pairs `(r, r + 16)`: ranks 0-15 run
  the conditional branch and ranks 16-31 the unconditional one, each over the whole step
  at batch 1, each prefilling only its own branch's prompt, with no K/V exchange. Inside
  the step graph the two branches' hidden states are all-gathered before the final layer,
  so every rank returns both predictions; the CFG combine and the scheduler run on the
  host. Against one TP32 group with both branches batched, a step went from 0.558 s to
  0.442 s (before the denoise attention kernel, below).
- **Within a replica** attention is head-parallel (2 of the 32 query heads per rank; each
  of the 8 KV heads shared by its 2 query ranks) and the MoE is expert-parallel (4 of the
  64 routed experts per rank, plus a 1/16 slice of the shared expert): ~10.5 GB of the
  24 GB core.
- **UNet ends.** The patch embed and final layer are split over all 32 ranks by GroupNorm
  group (`unet_parallel: true`).
- **VAE.** The decode runs on device in float16 with float32 reductions, split by image
  rows over ranks 0 and 1 (`vae_device: neuron`, `vae_device_dtype: float16`,
  `vae_parallel: 2`), on the backbone's own cores.

A request must use CFG (`guidance_scale > 1`) with this layout. The code also accepts
`tensor_parallel_size: 32` with `cfg_parallel_size: 1` (both branches batched on one TP32
group), but only the shipped config is validated.

## Usage

Set up the host or the DLC with the [setup guide](../getting-started/setup-guide.md). The
pipeline needs `libtorch-neuronx-lite>=2.11.0.1.0.2651` (pinned in
`requirements/core.txt`); older Lite builds lack the `_compiler` module the plugin uses.
On a fresh HyperPod `trn2.48xlarge`,
[`bootstrap_trn2_node.sh`](../../examples/hunyuan_image3/bootstrap_trn2_node.sh) pulls the
DLC, starts the container with persistent caches under `/opt/dlami/nvme/omni-lite`,
installs the plugin and the Lite pin and downloads the checkpoint at the validated
revision; [`device_bringup.sh`](../../examples/hunyuan_image3/device_bringup.sh) then runs
the stages `smoke` (2 layers, 2 steps), `full`, `serve` and `bench`.

Offline (writes `hunyuan_image3_output.png`):

```bash
python examples/hunyuan_image3/run.py --dev                       # 2 steps: compile and warm the graphs
python examples/hunyuan_image3/run.py \
  --model-path /models/HunyuanImage-3.0-Instruct --prompt "a red teapot on a wooden table"
python examples/hunyuan_image3/run.py --profile --runs 3          # timed warm requests
```

`--prefill-len`, `--moe-kernel nki|torch`, `--vae-device neuron|cpu` and
`--vae-parallel N` override the stage config for one run; `--num-layers N` truncates the
decoder stack for bring-up (the image is then meaningless).

Online (OpenAI-compatible chat API; the third argument defaults to
`hunyuan_image3_dit.yaml`):

```bash
bash examples/hunyuan_image3/serve.sh /models/HunyuanImage-3.0-Instruct 8091
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"a red teapot on a wooden table"}],
       "extra_body":{"height":1024,"width":1024,"num_inference_steps":50,
                     "guidance_scale":2.5,"seed":42}}' \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- | base64 -d > hunyuan_image3_output.png
```

[`benchmark.sh`](../../examples/hunyuan_image3/benchmark.sh) runs vLLM-Omni's
`diffusion_benchmark_serving.py` (from a vllm-omni source checkout) against the running
server: 1024x1024, 50 steps, concurrency 1.

**Compile cache.** The first request compiles the prompt-prefill and denoise-step graphs
and the device VAE's stage graphs; `run.py` and `serve.sh` raise vLLM-Omni's init and
handshake timeouts to 4 hours for it. Point `TORCH_NEURONX_NEFF_CACHE_DIR` at storage that
outlives the container so only the first run pays it. Inside the first request the split
VAE compiles 14 stage NEFFs (7 per rank, the two ranks in parallel, ~36 min); to fill them
beforehand, stop the server and run, from the same checkout the server imports:

```bash
python examples/hunyuan_image3/compile_vae_stages.py --vae-parallel 2   # the stage config's row split, cores 0-31
```

`--dtype` must match `vae_device_dtype` (float16). The cache key includes the source
path and line of every traced op, so a cache is reused only by the same files at the same
path.

Per-request stage timings are written to
`$WORKLOAD_OUTPUT_RW/metrics/pipeline_perf_metrics.json` (`/tmp/metrics/...` by default):
`prefill_seconds`, `denoise_seconds`, `seconds_per_step`, `vae_decode_seconds` and
`e2e_forward_seconds`.

## Results

Measured 2026-10-15 on one `trn2.48xlarge` with this build: `hunyuan_image3_dit.yaml`,
1024x1024, 50 steps, CFG 2.5, seed 42, the Mars-observatory prompt of the accuracy tests
(`run.py`'s default prompt). *Warm* requests follow a first request on a filled NEFF
cache.

| Measurement | How | Result |
| --- | --- | --- |
| Warm offline request | `run.py --profile --runs 2` (tier 3) | 19.36-19.39 s |
| Stages of a warm request | `pipeline_perf_metrics.json` | prefill 0.04 s, denoise 18.07 s (0.361 s/step), VAE decode 1.19 s |
| First request, NEFF cache filled | `run.py`, first generation | 151.5 s (weights and NEFFs loading) |
| Warm request over HTTP | `serve.sh` + `/v1/chat/completions` | 20.04-20.06 s, image byte-identical to the offline one |
| Tier 1, components | `test_hunyuan_image3_components_accuracy.py` | pass (3 tests, 6 min) |
| Tier 2, one denoising step | `test_hunyuan_image3_pipeline_accuracy.py` | pass (σ 0.990 / 0.982 / 1.11, 8 min) |
| Tier 3, end to end (`t2i`) | `test_hunyuan_image3_e2e_accuracy.py` | pass (SSIM 1.0, 5 min) |
| Against the build this one was cut from | sha256 of the Mars image | byte-identical (`685365e5...`), same token ids |

The tier-3 golden (`test/neuron/goldens/` on the accuracy branch, see
[Accuracy Evaluation](#accuracy-evaluation)) was regenerated from this build (`HY3_UPDATE_GOLDEN=1`) after its output was confirmed byte-identical to the
previous golden; only the stage-config hash, the provenance and the latency in
`manifest.json` changed. Tier 3 holds the faster of two warm requests within 10%
(`HY3_LATENCY_TOLERANCE`) of the manifest's `warm_seconds` (19.39 s).

## Accuracy Evaluation

The hardware accuracy tests are not part of this checkout: like the Wan2.2 suite the
[onboarding guide](../model-dev/onboarding-models.md) refers to, they belong with the
development repository's CI. The suite these results come from — `test/neuron/`, its
goldens and the run logs (`test/neuron/RESULTS.md`) — is on the branch
[`hunyuan-image3-instruct-accuracy`](https://github.com/AutoJunjie/vllm-omni-neuron/tree/hunyuan-image3-instruct-accuracy/test/neuron); the `test/neuron/` paths below refer to
it. It checks the port in three tiers, cheapest first. Tiers 1 and 2 compare
against the **official implementation** — the checkpoint's own
`modeling_hunyuan_image_3.py`, run on the host (`test/neuron/hy3_official.py`), which
shares no code with the plugin or with vLLM-Omni's port — using `assert_close_three_way`
(`vllm_neuron.accuracy.testing`): the official code in float32 (baseline) and in the
device's dtype (expected), against the device (actual). A comparison passes when the
device's error distribution matches the reference's own same-dtype error (σ-ratio <= 1, or
Bhattacharyya coefficient >= 0.99). Tier 3 is a regression check against a reviewed
previous Neuron output. Every test skips off-device (`require_neuron_device`) and reads
the checkpoint from `HY3_MODEL_PATH` (default `/workspace/models/HunyuanImage-3.0-Instruct`).

| Tier | What runs on device | Reference | Result on this build (2026-10-15) |
| --- | --- | --- | --- |
| 1. Components (`test_hunyuan_image3_components_accuracy.py`) | decoder layer 0 at TP16 (NKI attention, o-proj, MoE): the prompt prefill's K/V and a denoise step; the UNet patch embed and final layer split over 16 ranks | the official layer over the same tokens as one sequence, and the official modules, bf16 | pass: K σ 0.63, V σ 0.82, denoise step σ 1.000, patch embed σ 0.96, final layer σ 0.84 |
| | the device VAE decoder, float16 with float32 GroupNorm statistics: split by rows over ranks 0/1 (32-rank world) and whole on one core | the official VAE decode in float16 with float32 GroupNorm statistics | pass: split σ 1.02 (BC 0.99988), one core σ 0.97 (BC 0.99988) |
| 2. One denoising step (`test_hunyuan_image3_pipeline_accuracy.py`) | the whole pipeline at the stage config, step 0 of the Mars prompt, recorded with its templated inputs | the official model — RoPE, embeddings, all 32 layers, UNet ends, CFG operator and scheduler — replaying that step (`hy3_official_step.py`), bf16 | pass: CFG-batch prediction σ 0.990, guided prediction σ 0.982, updated latents σ 1.11 (BC 0.998) |
| 3. End to end (`test_hunyuan_image3_e2e_accuracy.py`, case `t2i`) | a full 50-step generation, two warm requests | the reviewed golden in `test/neuron/goldens/`, bound in `manifest.json` to its request, stage config, checkpoint, token ids, archived initial latents and environment; SSIM >= 0.95, latency within 10% | pass: SSIM 1.0000 (byte-identical), warm 19.36 / 19.37 s against 19.39 s |

Tier 2's request inputs (token ids, image slots, attention mask) come from the device's
step trace, i.e. from vLLM-Omni's port of the official template; everything computed from
them is the official code's. The SSIM floor was calibrated on the text-to-image output: a
rerun scores 1.0, sound layout or kernel changes 0.967-0.989, an unrelated image 0.18.
`HY3_UPDATE_GOLDEN=1` (with `HY3_MODEL_REVISION`) rewrites the golden and the manifest
from a reviewed run. These are manual runs on one `trn2.48xlarge`; the development
repository's CI manifest is not part of this checkout.

From a checkout of that branch:

```bash
pytest -s test/neuron/test_hunyuan_image3_components_accuracy.py   # ~6 min
pytest -s test/neuron/test_hunyuan_image3_pipeline_accuracy.py     # ~8 min, ~500 GB of host memory
pytest -s test/neuron/test_hunyuan_image3_e2e_accuracy.py          # ~5 min on a filled NEFF cache
```

The CPU unit tests in this checkout (`VLLM_NEURON_CPU_MODE=1 python
test/unit/hunyuan_image3/<file>`) check the port's math on tiny random checkpoints:
`test_numerics.py` (the prefill + denoise split
against one full-sequence forward of the reference math, 7.07e-08 relative — float32
round-off), `test_sharding.py` (every weight loader rank by rank at TP16 and TP32),
`test_cfg_parallel.py` (CFG replicas against the TP-only step), `test_unet_parallel.py`,
`test_vae_device.py`, `test_vae_nki_conv.py`, `test_vae_sharded.py`, and
`test_request_options.py` (what a request and a stage config may ask for).

## Known limitations

- **Text-to-image only.** A request with `bot_task` other than `image` / `auto` /
  `vanilla` / unset, with think or recaption text, or with conditioning images raises
  `NotImplementedError` before any device work.
- **1024x1024 only.** Any other size raises `ValueError`.
- **CFG required.** With `cfg_parallel_size: 2`, `guidance_scale <= 1` raises.
- **One request at a time.** Concurrent requests run serially (`max_batch_size: 1`).
- **One prompt bucket.** The prompt is right-padded to `prefill_len` (512). A longer
  prompt is an error, not a truncation: raise `prefill_len`, which compiles a new prefill
  and denoise graph.
- **Every step is computed.** There is no step caching in this build.
- **Unsupported `model_config` keys raise.** `step_cache`, `tea_cache`,
  `prefill_buckets`, `cfg_combine`, `tp_sequence_parallel`, `enable_ring_attention`,
  `vae_device_core`, `vae_encode_device` and `ar_max_new_tokens` are rejected when set, as
  is `ring_degree > 1`.
- **The split VAE decode** agrees across its ranks before its collectives, so a rank whose
  decoder cannot be built sends both to rank 0's host decode (~29 s); a failure inside a
  stage still leaves the other rank waiting in its collective.
- **Rejected requests return HTTP 500.** vLLM-Omni 0.24 wraps any exception from the
  diffusion worker as "Image generation failed", so an unsupported size or `bot_task`
  reaches the client as a 500 with the plugin's message, not as a 400.
- **No FP8.** BF16 only.

## Implementation notes

### Execution model

Upstream runs the first denoising step over `prompt + image tokens` and later steps over
`timestep + image tokens`, reusing a prompt KV cache. On Neuron the split is explicit and
each part compiles once:

1. **Prefill** — `forward_prefill(inputs_embeds, cos, sin)`: causal self-attention over
   the prompt, right-padded to `prefill_len`, returning the per-layer prompt K/V. Once per
   request.
2. **Denoise step** — one graph that fuses the patch embed (`UNetDown`), the 32-layer
   backbone and the final layer (`UNetUp`), consuming the prompt K/V plus the step's own
   K/V over 4097 tokens (timestep token + 4096 image tokens). Replayed every step.

The split is exact: the generation mask is lower-triangular with a full-attention block
over the generated image, so prompt tokens never attend to image tokens. The prompt's
padding columns are masked out of every step by additive key biases read straight out of
that mask; the timestep token and the image tokens get one bias each, because the mask
gives them different visibility. The scheduler, the CFG combine and the 133k x 4096 token
embedding stay on the host, which keeps a 1.1 GB table off every core and means no
uncompiled operation is dispatched to the device under the Lite runtime.

### NKI kernels

| Kernel | Where | Fallback |
| --- | --- | --- |
| `attention_cte`, causal, d-major output | prompt prefill attention | torch masked softmax |
| `attention_cte` with prefix caching | denoise attention of the image queries (`denoise_attention: nki`) | fused torch softmax (`denoise_attention: torch`) |
| `output_projection_cte` | every attention output projection | `matmul` + bias |
| `moe_cte`, `shard_on_i_hybrid`, block 512 | routed experts (`moe_kernel: nki`) | every local expert densely (`moe_kernel: torch`) |

Each kernel is gated on `can_run_kernel` and its tiling limits, so CPU mode and
fake-tensor tracing take the torch path; `attention_kernel: torch` turns off both
attention-side kernel families for bisection.

- **Denoise attention.** The right-padded prompt K/V is the kernel's prefix and
  `prior_used_len` — a runtime value, read off the image key bias — masks its padding, so
  prompts of any length share one NEFF; the step's own K/V is the non-causal active part.
  This needs every image token to see the whole prompt and the whole step, which the
  pipeline checks against the generation mask. The timestep token's one row stays in
  torch with its own bias. A step went from 0.442 s to 0.357 s at TP16 x CFG2.
- **MoE.** `shard_on_i_hybrid` with `num_static_block=0` runs a dynamic loop over only the
  blocks the router filled; `shard_on_block` walks every block, live or not. At one rank's
  TP32 denoise shapes during bring-up: 5.19 ms a layer at block 512, against 10.54 ms
  dense and 18.34 ms for `shard_on_block`. Step by step against float32 it is as close as
  the dense path (all within ±3% of each other, ~5.5e-02 relative on the combined CFG
  prediction — the bf16 floor). The padding mask is built as an `arange < live`
  comparison: built by in-graph slice assignment, it made `neuronx-cc` fail with
  `NCC_IMPR902 MaskPropagation`.

### UNet split

The patch embed and final layer are ResBlocks over the 64x64 latent grid with up to 4096
channels; run whole on every rank they were ~245 ms of a 760 ms TP32 step, the final layer
mostly in small DMAs turning the token tensor back into an image layout.
`unet_parallel.py` splits them by GroupNorm group — 32 groups, one per rank, so the
statistics stay local — with column-parallel convolutions over an all-gathered input and
the timestep projection producing only the rank's scale/shift. Measured at TP32 a step
went from 0.7585 s to 0.5578 s, and its step-wise error against float32 was no worse
(5.25e-02, against 5.64e-02 replicated).

### VAE decode on device

The whole decoder is 10,746,838 instructions at 1024x1024 against the
`NeuronHloVerifier` ceiling of 10,000,000, and spatial tiling makes the count worse, not
better. `vae_device.py` therefore splits it by depth into seven stage graphs (`in+mid`,
`up0`-`up4`, `out`) with intermediates kept on device, and `vae_nki_conv.py` routes the
3x3x3 convolutions through `nkilib`'s `conv3d`, splitting the 8192- and 4096-wide
`UpsampleDCAE` filter banks by output channel. GroupNorm statistics and the mid-block
softmax accumulate in float32 — without that a bfloat16 decode was garbage (1.03
relative error) — and in float16 each GroupNorm scales its group by a power of two before
squaring, so nothing past float16's range is formed. float16 decodes 66.4 dB PSNR from the
host float32 decode (bfloat16: 58.9 dB) at the same speed.

On one core the float16 decoder holds ~14.7 GiB and does not fit beside a TP16 backbone
shard. `vae_parallel: 2` splits it by image rows over ranks 0 and 1: each rank owns half
the rows of every resolution plus a halo (one row per 3x3 convolution from a cut edge,
mapped back through the upsamples), `in+mid` runs whole on both, the GroupNorms all-reduce
their per-rank sums so both normalise with the whole image's statistics, and the output
stage all-gathers the owned rows (`ShardedDeviceDecoder`). Against one core of its own:

| | One core of its own | `vae_parallel: 2` |
| --- | ---: | ---: |
| VAE decode | 1.87 s | **1.16 s** |
| Peak HBM of the busiest core | 14.7 GiB | 20.7 of 24 GiB (ranks 0/1: backbone + half the VAE) |
| Logical cores | 33 | **32** |

The split image is 69.7 dB PSNR from the one-core decode (max 1/255). `vae_device: cpu`
decodes on the host in float32 (`vae_dtype`), the reference, ~29 s; the device path falls
back to it for a request whose device decode fails.

### Runtime

On the Lite runtime the worker pins the CCOM root (`NEURON_RT_ROOT_COMM_ID`) to
`localhost:61234 + <first core of the stage>`, outside the kernel's ephemeral port range;
left to pick a free port, rank 0 could lose it to one of its own sockets and every other
rank waited forever. `env_profiles.HUNYUAN_IMAGE3` is Wan2.2's runtime profile with a
14400 s compilation budget and `NEURON_RT_EXEC_TIMEOUT=600`.

## Troubleshooting

### Workers hang at startup, or the stage fails to bind its root port

- **Possible solution**: Two servers on one node need disjoint `devices` ranges, since the
  CCOM root port is keyed on the stage's first core. Leave `NEURON_RT_VISIBLE_CORES` unset
  (`serve.sh` unsets it): vllm-neuron rejects it under multiprocessing.

### Every start is a cold compile

- **Possible solution**: The NEFF cache key includes the absolute path and line of every
  traced op. Run from the same checkout at the same path as the cached build, don't edit
  lines above a traced function, and persist `TORCH_NEURONX_NEFF_CACHE_DIR`.

### A request or the stage config is rejected at start

- **Possible solution**: `Prompt needs N tokens but the prompt bucket is 512`: raise
  `prefill_len` (`run.py --prefill-len`; one more compile). `model_config.<key> ... is not
  supported in this build`: remove the key (see
  [Not in this build](#not-in-this-build--roadmap)). `guidance_scale ... turns CFG off`:
  use a `guidance_scale` above 1.

### The VAE decode takes ~29 s instead of ~1.2 s

- **Possible solution**: The log says `device VAE unavailable` or `device VAE decode
  failed; using the host`. Check the cores' HBM with `neuron-monitor` (ranks 0/1 peak at
  20.7 of 24 GiB) and that `vae_device_dtype` matches the dtype the cache was filled with.

### `libtorch_neuronx_lite` has no `_compiler` module

- **Possible solution**: The DLC's bundled Lite build is too old; install
  `libtorch-neuronx-lite>=2.11.0.1.0.2651` (the bootstrap script pins
  `2.11.0.1.0.2651+723ba691`).

## Not in this build / roadmap

Developed on the predecessor build and planned for later milestones; none is in this
build's code or stage config:

- **Think / recaption** (`bot_task: think`, `recaption`, `think_recaption`): AR decoding
  on the DiT's own cores and weights before the image.
- **Image editing / IT2I**: device VAE encoder, SigLIP2 and aligner conditioning tokens,
  bounded prompt prefill.
- **Step caching**: Cache-DiT DBCache and TeaCache.
- **`NeuronCFGParallelMixin`** as an alternative CFG combine.
- **Megatron sequence parallelism** (`tp_sequence_parallel`).
- **Token-split context parallelism** (`ring_degree: 2`) for requests without CFG.
- **vLLM-Omni's tile-parallel VAE** (`vae_patch_parallel_size`).
- **The VAE decode on a separate core** (`vae_device_core`).
- **Multiple prompt buckets** (`prefill_buckets`) and **other resolutions**.
- **The 64-core configuration** experiment.
- **Benchmark, kernel-check and GenEval scripts**, and a GenEval score for this build.

## Related information

- [Onboarding a model](../model-dev/onboarding-models.md) — the process this
  implementation follows.
- [Evaluating and debugging model accuracy](../model-dev/accuracy-evaluation-debugging.md)
  — the three-way comparison behind tiers 1 and 2.
- [Setup guide](../getting-started/setup-guide.md) — environment and container.
- [Wan2.2-T2V-A14B model card](wan22-t2v-14b.md) — the reference blueprint.
