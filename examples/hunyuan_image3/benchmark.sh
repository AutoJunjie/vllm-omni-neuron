#!/usr/bin/env bash
# Run vLLM-Omni's diffusion serving benchmark against the Neuron HunyuanImage-3.0 server.
#
#   bash examples/hunyuan_image3/benchmark.sh <VLLM_OMNI_SRC> [PORT] [NUM_PROMPTS]
#
# `VLLM_OMNI_SRC` is a vllm-omni source checkout: the benchmark lives at
# benchmarks/diffusion/diffusion_benchmark_serving.py and is not part of the installed
# wheel. The server must already be serving (see serve.sh); start it first, let the cold
# NEFF compile finish, and warm it with one request so the benchmark measures steady state.
set -euo pipefail

OMNI_SRC=${1:?usage: benchmark.sh <VLLM_OMNI_SRC> [PORT] [NUM_PROMPTS]}
PORT=${2:-8091}
NUM_PROMPTS=${3:-4}
OUT=${BENCH_OUT:-hunyuan_image3_benchmark.json}

BENCH="$OMNI_SRC/benchmarks/diffusion/diffusion_benchmark_serving.py"
[ -f "$BENCH" ] || { echo "benchmark script not found: $BENCH" >&2; exit 1; }

python "$BENCH" \
  --endpoint /v1/chat/completions \
  --backend chat \
  --dataset random \
  --task t2i \
  --host 127.0.0.1 \
  --port "$PORT" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency 1 \
  --random-request-config '[{"width":1024,"height":1024,"num_inference_steps":50,"weight":1}]' \
  --output-file "$OUT"

echo "wrote $OUT"
