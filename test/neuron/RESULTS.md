# HunyuanImage-3.0-Instruct: hardware accuracy results

The three accuracy tiers of the onboarding guide (Step 4) for the HunyuanImage-3.0-Instruct
pipeline, run by hand on one `trn2.48xlarge`. This branch is the pull request's commits
plus this directory; the tests and goldens are not part of the pull request (the release
checkout carries no hardware tests). The raw pytest output of every run is in
[`logs/`](logs/).

## 2026-10-17, on the pull request's final code

Code: `b434183`, the pull request's commits as of that morning (`3165a00`) with this
directory on top. The pull request's head (`12e91ab`) changes only documentation and
example scripts after it (the same `geneval_generate.py` drove the GenEval run), so the
model code is the same `vllm_omni_neuron` tree `e2de2527` that `goldens/manifest.json`
records. Logs: `logs/20261017_*`.

| Tier | Comparison | dtype | σ-ratio | BC | Result |
| --- | --- | --- | ---: | ---: | --- |
| 1 | layer 0 prefill K | bf16 | 0.630 | 0.9834 | pass |
| 1 | layer 0 prefill V | bf16 | 0.821 | 0.9904 | pass |
| 1 | layer 0 denoise step (4097 tokens) | bf16 | 1.000 | 1.0000 | pass |
| 1 | UNet patch embed, split over 16 ranks | bf16 | 0.960 | 0.9986 | pass |
| 1 | UNet final layer, split over 16 ranks | bf16 | 0.835 | 0.9922 | pass |
| 1 | VAE decode, rows split over ranks 0/1 | fp16 | 1.024 | 0.9999 | pass |
| 1 | VAE decode, one core | fp16 | 0.972 | 0.9999 | pass |
| 2 | CFG-batch prediction, step 0 | bf16 | 0.990 | 0.9999 | pass |
| 2 | guided prediction, step 0 | bf16 | 0.982 | 0.9997 | pass |
| 2 | scheduler-updated latents, step 0 | bf16 | 1.113 | 0.9979 | pass |

Tier 1 compares against the checkpoint's own `modeling_hunyuan_image_3.py` on the host
(`hy3_official.py`) through `assert_close_three_way` (float32 baseline, device-dtype
expected, device actual). Tier 2 records step 0 of the Mars prompt on the device and
replays it through the whole official model on the host (`hy3_official_step.py`). The
one-core VAE case ran last, on its own: its graphs were not in the NEFF cache (the 10-16 run had been stopped
during this compile), so the test compiled them and then passed, 24 min in all
(`logs/20261017_tier1_vae_one_core.log`).

Tier 3 (`test_hunyuan_image3_e2e_accuracy.py`): full 50-step 1024x1024 generations, two
warm requests each, against the reviewed goldens in [`goldens/`](goldens/) (SSIM >= 0.95;
the faster warm request within 10% of the manifest's `warm_seconds`, 25% for editing):

| Case | SSIM | Warm requests | Target |
| --- | --- | --- | --- |
| `t2i` | 1.0000 | 19.34 / 19.35 s | 19.41 s |
| `edit` | 1.0000 | 28.78 / 24.06 s | 28.6 s (+25%) |
| `t2i_think` | 1.0000 | 28.45 / 28.52 s | 28.54 s |

CPU unit tests, `VLLM_NEURON_CPU_MODE=1 python -m pytest test/unit/hunyuan_image3`: 50
passed (`logs/20261017_unit.log`).

GenEval over all 553 prompts on the same code (one image per prompt, seed 0): see the
model card's Accuracy Evaluation; 0.864 (single object 98.75%, two objects 98.99%,
counting 72.50%, colors 88.30%, position 80.00%, color attribution 80.00%), every one of
the 553 images byte-identical to the 2026-10-15 run of the same configuration. The
summary and the per-image results are `logs/20261017_geneval_summary.txt` and
`logs/20261017_geneval_results.jsonl`; HTTP latency and the serving benchmark are
`logs/20261017_http.txt` and `logs/20261017_benchmark_*.log`.

## Earlier runs

The same tiers on earlier states of the pull request, kept for the record:

| Date | Code (`hunyuan_image3` model tree) | Tier 1 | Tier 2 | Tier 3 | Logs |
| --- | --- | --- | --- | --- | --- |
| 2026-10-15 | M1 before its final clean-up (`6e601538`) | 3 tests, all 7 comparisons pass, same values | pass, same values | `t2i` SSIM 1.0, 19.36 / 19.37 s | `logs/20261015_*` |
| 2026-10-16 | M1 (`5ad8bdb0`), CCOM pin as a plain assignment | 2 tests (6 comparisons) pass; the one-core VAE was stopped by hand during a cold compile (SIGTERM, `exited -15`) | pass, same values | `t2i` SSIM 1.0, 19.31 / 19.39 s | `logs/20261016_m1_*` |
| 2026-10-16 | M2 (`e2de2527` package tree) | — | — | goldens registered (`HY3_UPDATE_GOLDEN=1`) after the three outputs were confirmed byte-identical to the development tree's, then checked: all SSIM 1.0 | `logs/20261016_m2_*` |

## Running

```bash
export HY3_MODEL_PATH=/path/to/HunyuanImage-3.0-Instruct
pytest -s test/neuron/test_hunyuan_image3_components_accuracy.py   # ~6 min on a filled NEFF cache
pytest -s test/neuron/test_hunyuan_image3_pipeline_accuracy.py     # ~8 min, ~500 GB of host memory
pytest -s test/neuron/test_hunyuan_image3_e2e_accuracy.py          # ~18 min on a filled NEFF cache
```
