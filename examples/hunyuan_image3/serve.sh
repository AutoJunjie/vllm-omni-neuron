#!/usr/bin/env bash
# Serve HunyuanImage-3.0-Instruct through the OpenAI-compatible chat API on Neuron.
#
#   bash examples/hunyuan_image3/serve.sh [MODEL_PATH] [PORT] [STAGE_CONFIG]
#
# Generate an image once the server is up:
#
#   curl -s http://localhost:8091/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"messages":[{"role":"user","content":"a red teapot on a wooden table"}],
#          "extra_body":{"height":1024,"width":1024,"num_inference_steps":50,
#                        "guidance_scale":2.5,"seed":42}}' \
#     | jq -r '.choices[0].message.content[0].image_url.url' \
#     | cut -d',' -f2- | base64 -d > hunyuan_image3_output.png
set -euo pipefail

MODEL=${1:-/models/HunyuanImage-3.0-Instruct}
PORT=${2:-8091}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE=${3:-$HERE/hunyuan_image3_stage.yaml}

# vllm-neuron rejects NEURON_RT_VISIBLE_CORES under multiprocessing; the stage config's
# `devices` range is what assigns a core to each worker.
unset NEURON_RT_VISIBLE_CORES

export TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
export VLLM_SLEEP_WHEN_IDLE=1
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_SCRATCHPAD_PAGE_SIZE=2048
export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=2048
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=167772160
export NEURON_CC_FLAGS="-O1 --hbm-scratchpad-page-size=2048"
export VLLM_NEURON_BACKEND=${VLLM_NEURON_BACKEND:-neuron_native}
export VLLM_NEURON_LIBTORCH_NEURONX_LITE=${VLLM_NEURON_LIBTORCH_NEURONX_LITE:-1}
export VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1
export VLLM_NEURON_COMPILATION_TIMEOUT=${VLLM_NEURON_COMPILATION_TIMEOUT:-14400}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}

exec vllm serve "$MODEL" \
  --omni \
  --host 127.0.0.1 \
  --port "$PORT" \
  --stage-configs-path "$STAGE" \
  --stage-init-timeout 14400 \
  --init-timeout 14400
