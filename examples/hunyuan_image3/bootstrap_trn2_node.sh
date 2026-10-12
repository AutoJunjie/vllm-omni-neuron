#!/usr/bin/env bash
# Bring up the omni-lite environment for the HunyuanImage-3.0 Neuron pipeline on a
# fresh trn2.48xlarge HyperPod node. Idempotent: each stage skips when already done.
#
# Runs on the node host (not inside the container). Logs to
# /opt/dlami/nvme/omni-lite/logs/bootstrap.log; the weight download logs to weights.log.
set -u
ROOT=/opt/dlami/nvme/omni-lite
IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04
NS="--namespace omni-lite"
BRANCH=feat/hunyuan-image3-neuron-lite
# The DLC ships libtorch-neuronx-lite 2.11.0.1.0.1284, which has no `_compiler`
# submodule; the plugin's lite_compat needs it for device_count/nki_op/mesh registry.
LITE_PIN="libtorch-neuronx-lite==2.11.0.1.0.2651+723ba691"
MODEL_REVISION=2ec2c78bee7d4b94157341fba86c4c2c7b1858b2

mkdir -p "$ROOT"/{logs,out,neff-cache,compile_dir,huggingface,vllm-cache,nki-cache,models}
exec >> "$ROOT/logs/bootstrap.log" 2>&1
echo "=== bootstrap $(date -u +%FT%TZ) ==="

stage() { echo "--- $* ---"; }

stage image
if ! nerdctl $NS images 2>/dev/null | grep -q pytorch-inference-vllm-neuronx; then
  nerdctl $NS pull "$IMAGE" || exit 1
fi

stage container
if ! nerdctl $NS ps 2>/dev/null | grep -q omni-lite; then
  nerdctl $NS rm -f omni-lite >/dev/null 2>&1
  DEVS=""
  for i in $(seq 0 15); do DEVS="$DEVS --device /dev/neuron$i"; done
  # NEURON_RT_VISIBLE_CORES is deliberately NOT set: vllm-neuron rejects it under
  # multiprocessing and assigns each worker a core from the stage config's range.
  nerdctl $NS run -d --name omni-lite \
    --entrypoint /bin/bash --workdir /workspace \
    --network host --ipc host --shm-size 64g \
    --ulimit memlock=-1 --ulimit nofile=65535:65535 \
    --cap-add SYS_ADMIN --cap-add IPC_LOCK \
    $DEVS \
    -e VLLM_OMNI_HOME=/workspace \
    -e VLLM_NEURON_BACKEND=neuron_native \
    -e VLLM_NEURON_LIBTORCH_NEURONX_LITE=1 \
    -e VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1 \
    -e HF_HOME=/workspace/huggingface \
    -e VLLM_CACHE_ROOT=/workspace/vllm-cache \
    -e NKI_COMPILE_CACHE_URL=/workspace/nki-cache \
    -e TORCH_NEURONX_NEFF_CACHE_DIR=/workspace/neff-cache \
    -v "$ROOT":/workspace \
    "$IMAGE" -c "sleep infinity" || exit 1
  sleep 3
fi

stage plugin
# Note the cd / before the install guard: the repo root is on sys.path whenever cwd is
# the checkout, so `import vllm_omni_neuron` there succeeds even with nothing installed.
# vllm_omni is the real signal — it only appears once the plugin's deps are installed.
nerdctl $NS exec omni-lite bash -lc '
set -e
if [ ! -d /workspace/plugin/.git ]; then
  rm -rf /workspace/plugin
  git clone -q https://github.com/AutoJunjie/vllm-omni-neuron.git /workspace/plugin
fi
cd /workspace/plugin
git fetch -q origin
git reset -q --hard origin/'"$BRANCH"'
git log --oneline -1
cd /
python -c "import vllm_omni" 2>/dev/null || \
  pip install -q --extra-index-url=https://pip.repos.neuron.amazonaws.com -e /workspace/plugin
pip install -q --extra-index-url=https://pip.repos.neuron.amazonaws.com "'"$LITE_PIN"'"
python - <<VERIFY
import torch, libtorch_neuronx_lite, vllm, vllm_neuron, vllm_omni
from vllm_omni_neuron import lite_compat
from vllm_omni_neuron.platform import NeuronOmniPlatform
print("devices", NeuronOmniPlatform.get_device_count(), "target", lite_compat.get_platform_target())
VERIFY
' || exit 1

stage weights
# huggingface_hub lives in the container, not on the host image, so the download runs
# there. ~160 GB of safetensors at a pinned revision.
if [ ! -f "$ROOT/models/HunyuanImage-3.0-Instruct/model.safetensors.index.json" ] &&
   ! pgrep -f hf_download_instruct >/dev/null; then
  cat > "$ROOT/hf_download_instruct.py" <<DOWNLOAD
from huggingface_hub import snapshot_download

snapshot_download(
    "tencent/HunyuanImage-3.0-Instruct",
    revision="$MODEL_REVISION",
    local_dir="/workspace/models/HunyuanImage-3.0-Instruct",
    ignore_patterns=["assets/*"],
    max_workers=16,
)
print("WEIGHTS_DONE")
DOWNLOAD
  setsid nohup nerdctl $NS exec omni-lite python /workspace/hf_download_instruct.py \
    > "$ROOT/logs/weights.log" 2>&1 &
  echo "started weight download in container (pid $!)"
fi

echo "BOOTSTRAP_OK $(date -u +%FT%TZ)"
