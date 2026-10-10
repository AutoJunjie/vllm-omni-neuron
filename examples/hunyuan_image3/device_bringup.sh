#!/usr/bin/env bash
# Staged on-device bring-up for the HunyuanImage-3.0 Neuron pipeline.
#
# Runs on the node host. Each stage writes its own log under
# /opt/dlami/nvme/omni-lite/logs and appends a one-line verdict to stages.log, so a
# long unattended run can be polled cheaply with `tail stages.log`.
#
#   bash device_bringup.sh [first_stage]
#
# Stages: nki, smoke, full, serve, bench
set -u
ROOT=/opt/dlami/nvme/omni-lite
NS="--namespace omni-lite"
LOGS=$ROOT/logs
STAGES=$LOGS/stages.log
MODEL=/workspace/models/HunyuanImage-3.0-Instruct
# vllm-omni commit whose benchmarks match the installed 0.24.0 wheel.
OMNI_COMMIT=d4a869f

mkdir -p "$LOGS" "$ROOT/out"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STAGES"; }

# Run a command in the container with the Neuron env the pipeline expects.
cexec() {
  nerdctl $NS exec omni-lite bash -lc "
    unset NEURON_RT_VISIBLE_CORES
    export NEURON_LOGICAL_NC_CONFIG=2
    export NEURON_SCRATCHPAD_PAGE_SIZE=2048
    export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=2048
    export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=167772160
    export NEURON_CC_FLAGS='-O1 --hbm-scratchpad-page-size=2048'
    export TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
    export TORCH_NEURONX_NEFF_CACHE_DIR=/workspace/neff-cache
    export TORCH_NEURONX_DEBUG_DIR=/workspace/compile_dir
    export WORKLOAD_OUTPUT_RW=/workspace/out
    cd /workspace/plugin
    $1"
}

# Device health after a stage: a wedged core shows up here before the next stage
# builds on top of it.
health() {
  nerdctl $NS exec omni-lite bash -lc 'neuron-ls --topology 2>&1 | head -5' \
    >> "$LOGS/health.log" 2>&1
  dmesg 2>/dev/null | grep -i neuron | tail -5 >> "$LOGS/health.log"
  echo "--- $(date -u +%FT%TZ)" >> "$LOGS/health.log"
}

stage_nki() {
  # One kernel per process: if a process wedges, the kernel on its command line is
  # the one that did it. The SwiGLU MLP kernel is deliberately excluded — it hung a
  # core on 2026-10-09 and is opt-in until that is understood.
  for kernel in o_proj attention; do
    say "nki $kernel: start"
    cexec "timeout 900 python examples/hunyuan_image3/check_nki_kernels.py \
      --tp-size 32 --only $kernel" > "$LOGS/nki_$kernel.log" 2>&1
    say "nki $kernel: rc=$? ($(grep -c OK "$LOGS/nki_$kernel.log") ok-lines)"
    health
  done
}

stage_smoke() {
  # 2 decoder layers, 2 denoising steps: exercises weight loading, both compiled
  # graphs, device execution, the host loop and the VAE decode in minutes. The
  # image is meaningless by construction.
  say "smoke: start"
  cexec "python - <<'PY'
import yaml
cfg = yaml.safe_load(open('examples/hunyuan_image3/hunyuan_image3_stage.yaml'))
mc = cfg['stage_args'][0]['engine_args']['model_config']
mc['num_layers'] = 2
open('/workspace/smoke_stage.yaml', 'w').write(yaml.safe_dump(cfg))
PY
timeout 5400 python examples/hunyuan_image3/run.py \
  --model-path $MODEL --num-inference-steps 2 \
  --stage-config /workspace/smoke_stage.yaml \
  --output /workspace/out/smoke.png" > "$LOGS/smoke.log" 2>&1
  say "smoke: rc=$?"
  health
}

stage_full() {
  say "full: start (cold compile)"
  cexec "timeout 28800 python examples/hunyuan_image3/run.py \
    --model-path $MODEL --num-inference-steps 50 \
    --output /workspace/out/hunyuan_image3.png" > "$LOGS/full.log" 2>&1
  say "full: rc=$? $(ls -l $ROOT/out/hunyuan_image3.png 2>/dev/null | awk '{print $5}') bytes"
  health
}

stage_serve() {
  say "serve: start"
  nerdctl $NS exec -d omni-lite bash -lc "
    unset NEURON_RT_VISIBLE_CORES
    cd /workspace/plugin
    export TORCH_NEURONX_NEFF_CACHE_DIR=/workspace/neff-cache
    export WORKLOAD_OUTPUT_RW=/workspace/out
    bash examples/hunyuan_image3/serve.sh $MODEL 8091 \
      > /workspace/logs/serve.log 2>&1"
  for _ in $(seq 1 240); do
    sleep 30
    if curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1; then
      say "serve: healthy"
      break
    fi
  done
  curl -sf http://127.0.0.1:8091/health > "$LOGS/serve_health.log" 2>&1 || {
    say "serve: never became healthy"
    return 1
  }
  say "openai request: start"
  curl -s http://127.0.0.1:8091/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"A cinematic photo of a glass observatory on Mars at sunrise, volumetric light, ultra detailed"}],"extra_body":{"height":1024,"width":1024,"num_inference_steps":50,"guidance_scale":2.5,"seed":42}}' \
    > "$LOGS/openai_response.json" 2>&1
  python3 - <<'PY' >> "$STAGES" 2>&1
import base64, json, pathlib
raw = pathlib.Path("/opt/dlami/nvme/omni-lite/logs/openai_response.json").read_text()
doc = json.loads(raw)
url = doc["choices"][0]["message"]["content"][0]["image_url"]["url"]
data = base64.b64decode(url.split(",", 1)[1])
out = pathlib.Path("/opt/dlami/nvme/omni-lite/out/openai_api.png")
out.write_bytes(data)
print(f"openai request: wrote {out} ({len(data)} bytes)")
PY
  health
}

stage_bench() {
  say "bench: start"
  nerdctl $NS exec omni-lite bash -lc "
    set -e
    if [ ! -d /workspace/vllm-omni-src/.git ]; then
      git clone -q --filter=blob:none --no-checkout \
        https://github.com/vllm-project/vllm-omni.git /workspace/vllm-omni-src
      cd /workspace/vllm-omni-src
      git sparse-checkout init --cone
      git sparse-checkout set benchmarks
      git checkout -q $OMNI_COMMIT
    fi
    ls /workspace/vllm-omni-src/benchmarks/diffusion/" >> "$LOGS/bench.log" 2>&1
  cexec "bash examples/hunyuan_image3/benchmark.sh /workspace/vllm-omni-src 8091 4" \
    >> "$LOGS/bench.log" 2>&1
  say "bench: rc=$?"
}

FIRST=${1:-nki}
RUN=0
for s in nki smoke full serve bench; do
  [ "$s" = "$FIRST" ] && RUN=1
  [ "$RUN" = "1" ] || continue
  "stage_$s"
done
say "device_bringup: done"
