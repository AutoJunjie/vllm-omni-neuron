# SPDX-License-Identifier: Apache-2.0
"""Run the staged device VAE decode in its own process, on a NeuronCore of its own.

The decoder does not share well. On the output rank's core it sits next to that rank's
backbone shard (7.7 GiB), and the runtime keeps far more resident than its 1.6 GiB of
weights: the seven stage NEFFs' code and DMA rings (4.5 GiB), a shared scratchpad sized
by the largest stage (5.75 GiB in float16), and an output buffer per stage that stays
allocated between requests (2.8 GiB, held below PyTorch -- ``empty_cache`` does not
return it). bfloat16 fits in the 24 GiB with little to spare; float16, whose ``up3``
stage compiles to 0.75 GiB more scratchpad, fails its second request with
``nrt_tensor_allocate`` once all seven stages are resident.

So the output rank starts this module as a child process pinned to a free core
(``model_config.vae_device_core``) -- a core outside the stage's ``devices`` range, which
a TP32 deployment leaves idle -- and exchanges latents and decoded frames with it over a
pair of pipes. The child loads only the decoder from the checkpoint and builds the same
:class:`~.vae_device.StagedDeviceDecoder`, so it traces the same graphs and hits the
same NEFFs as the in-process path. Each request costs one ~0.5 MB transfer in and one
~12 MB transfer out.

The child exits when its request pipe closes, so it cannot outlive the worker that
started it.
"""

import argparse
import io
import json
import logging
import os
import struct
import subprocess
import sys
import time

import torch

logger = logging.getLogger(__name__)

# The worker's distributed identity and device range. The child is not part of the
# process group, and inheriting these points its runtime at the group's rendezvous; the
# stage's NEURON_VISIBLE_DEVICES range makes the Lite runtime reject device index 0 on a
# process that is not rank 0. NEURON_RT_VISIBLE_CORES alone pins the child's core.
_DISTRIBUTED_ENV = (
    "NEURON_VISIBLE_DEVICES",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NEURON_RT_ROOT_COMM_ID",
)


def _send(stream, message: dict) -> None:
    buffer = io.BytesIO()
    torch.save(message, buffer)
    payload = buffer.getvalue()
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _recv(stream) -> dict:
    header = stream.read(8)
    if len(header) < 8:
        raise EOFError("VAE worker pipe closed")
    (length,) = struct.unpack("<Q", header)
    return torch.load(io.BytesIO(stream.read(length)), weights_only=True)


class RemoteDeviceDecoder:
    """Client for a VAE decode worker on another core. Same interface as the in-process
    :class:`~.vae_device.StagedDeviceDecoder`: ``supports(latents)`` and
    ``decode(latents)``, which returns float32 on the host."""

    def __init__(self, model_path: str, vae_config, core: int, dtype: str):
        from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_nki_conv import _MAX_SPATIAL

        self.core = core
        self.ffactor_spatial = int(vae_config["ffactor_spatial"])
        self.max_spatial = _MAX_SPATIAL

        start = time.perf_counter()
        request_read, request_write = os.pipe()
        result_read, result_write = os.pipe()
        env = {key: value for key, value in os.environ.items() if key not in _DISTRIBUTED_ENV}
        env["NEURON_RT_VISIBLE_CORES"] = str(core)
        command = [sys.executable, "-m", __name__, "--model-path", model_path]
        command += ["--dtype", dtype, "--in-fd", str(request_read), "--out-fd", str(result_write)]
        self._process = subprocess.Popen(
            command, env=env, pass_fds=(request_read, result_write), stdin=subprocess.DEVNULL
        )
        os.close(request_read)
        os.close(result_write)
        self._requests = os.fdopen(request_write, "wb")
        self._results = os.fdopen(result_read, "rb")

        ready = self._receive()
        logger.info(
            "HunyuanImage3: VAE decode worker pid %d on core %d: %s, ready in %.1fs",
            self._process.pid,
            core,
            ready["info"],
            time.perf_counter() - start,
        )

    def _receive(self) -> dict:
        try:
            message = _recv(self._results)
        except EOFError:
            code = self._process.wait(timeout=30)
            raise RuntimeError(f"VAE decode worker on core {self.core} exited ({code})") from None
        if "error" in message:
            raise RuntimeError(f"VAE decode worker on core {self.core}: {message['error']}")
        return message

    def supports(self, latents: torch.Tensor) -> bool:
        """Same spatial limit as the in-process decoder (see ``StagedDeviceDecoder``)."""
        height, width = latents.shape[-2:]
        return max(height, width) * self.ffactor_spatial <= self.max_spatial

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        _send(self._requests, {"latents": latents.detach().float().contiguous()})
        return self._receive()["decoded"]

    def close(self) -> None:
        """Close the request pipe; the worker exits on EOF."""
        if self._process.poll() is None:
            self._requests.close()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001  interpreter teardown
            pass


def _load_decoder(model_path: str, dtype: torch.dtype):
    """Build the decoder-only VAE on the host and load its weights from the checkpoint."""
    from safetensors import safe_open
    from vllm.transformers_utils.config import get_config
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    vae_config = dict(get_config(model_path, trust_remote_code=True).vae)
    vae = AutoencoderKLConv3D.from_config({**vae_config, "only_decoder": True}).eval()
    vae.requires_grad_(False)
    with open(os.path.join(model_path, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    targets = {f"vae.{name}": name for name in vae.state_dict()}
    files: dict[str, list[str]] = {}
    for key in targets:
        if key in weight_map:
            files.setdefault(weight_map[key], []).append(key)
    state = {}
    for file_name, keys in files.items():
        with safe_open(os.path.join(model_path, file_name), framework="pt", device="cpu") as f:
            for key in keys:
                state[targets[key]] = f.get_tensor(key).to(dtype)
    missing = sorted(set(targets.values()) - set(state))
    if missing:
        raise RuntimeError(f"VAE decoder weights missing from the checkpoint: {missing[:6]}")
    vae.load_state_dict(state, strict=False, assign=True)
    return vae, vae_config


def _serve(args) -> None:
    import vllm_omni_neuron.bootstrap  # noqa: F401  must precede vllm imports

    torch.nn.functional.gelu = torch.ops.aten.gelu.default

    from vllm_omni.platforms import current_omni_platform

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import StagedDeviceDecoder
    from vllm_omni_neuron.lite_compat import ensure_current_device_index, initialize

    requests = os.fdopen(args.in_fd, "rb")
    results = os.fdopen(args.out_fd, "wb")
    try:
        dtype = getattr(torch, args.dtype)
        initialize()
        ensure_current_device_index()
        device = current_omni_platform.get_torch_device(0)
        host_vae, vae_config = _load_decoder(args.model_path, dtype)
        decoder = StagedDeviceDecoder(host_vae, vae_config, device, dtype=dtype)
        del host_vae
    except Exception as error:  # noqa: BLE001  reported to the client, which raises
        _send(results, {"error": f"startup failed: {error!r}"})
        raise
    _send(results, {"info": f"{args.dtype} on {device}"})

    while True:
        try:
            message = _recv(requests)
        except EOFError:
            return
        try:
            _send(results, {"decoded": decoder.decode(message["latents"])})
        except Exception as error:  # noqa: BLE001  one bad request must not kill the worker
            logger.exception("VAE decode failed")
            _send(results, {"error": repr(error)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 VAE decode worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--in-fd", type=int, required=True)
    parser.add_argument("--out-fd", type=int, required=True)
    _serve(parser.parse_args())
