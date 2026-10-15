# SPDX-License-Identifier: Apache-2.0
"""Tier 1: the HunyuanImage-3.0 components on device against the official code, three ways.

Each component runs three times on identical inputs: the official implementation (the
checkpoint's own modeling code, :mod:`hy3_official`) on the host in float32 (baseline)
and in the device's dtype (expected: the dtype's own rounding), and the plugin on the
device (actual). ``assert_close_three_way`` (``vllm_neuron.accuracy.testing``) passes
when the device's error distribution matches the reference's same-dtype error --
isolating what the port, the Neuron kernels and compilation add from what the dtype
costs anyway.

- **Decoder layer** (checkpoint layer 0): the prompt prefill's K/V and one denoise step
  at the default config's TP16 sharding, compiled, with the NKI attention, o-proj and
  MoE kernels, on 16 cores; the reference runs the official layer over the same tokens
  as one sequence. bfloat16.
- **Patch embed / final layer**: split by GroupNorm group over the 16 ranks
  (``unet_parallel``) on device, the official modules whole on the host. bfloat16.
- **VAE decoder**: the staged device decoder (float16, float32 GroupNorm reductions),
  split by rows over ranks 0 and 1 as the default stage config runs it
  (``vae_parallel: 2``, 32-rank world) and whole on one core (``vae_parallel: 1``),
  against the official decode -- in float16 with float32 GroupNorm statistics.

    pytest -s test/neuron/test_hunyuan_image3_components_accuracy.py
"""

import os
import sys

import pytest
import torch
from hy3_helpers import run, start, wait

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "hy3_component_worker.py")
TP = 16
WORLD = 32  # the default stage config's ranks
PREFILL, PROMPT, STEP, HIDDEN, HALF = 512, 300, 4097, 4096, 64


def _inputs(path) -> None:
    g = torch.Generator().manual_seed(0)
    cos = lambda n: torch.rand(1, n, HALF, generator=g) * 2 - 1  # noqa: E731
    prompt_cos, step_cos = cos(PREFILL), cos(STEP)
    timestep_bias = torch.full((1, 1, 1, PREFILL + STEP), -1.0e30)
    image_bias = timestep_bias.clone()
    for bias in (timestep_bias, image_bias):
        bias[..., :PROMPT] = 0.0
    timestep_bias[..., PREFILL] = 0.0  # the timestep token sees the prompt and itself
    image_bias[..., PREFILL:] = 0.0  # image tokens see the prompt and the whole step
    torch.save(
        {
            "prompt_embeds": (torch.randn(1, PREFILL, HIDDEN, generator=g) * 0.05).bfloat16(),
            "prompt_cos": prompt_cos, "prompt_sin": torch.sqrt(1 - prompt_cos**2),
            "step_hidden": torch.randn(1, STEP, HIDDEN, generator=g).bfloat16(),
            "step_cos": step_cos, "step_sin": torch.sqrt(1 - step_cos**2),
            "timestep_key_bias": timestep_bias, "image_key_bias": image_bias,
            "latents": torch.randn(1, 32, 64, 64, generator=g).bfloat16(),
            "timestep": torch.tensor([800.0]),
            "unet_tokens": torch.randn(1, 64 * 64, HIDDEN, generator=g).bfloat16(),
            "vae_latents": torch.randn(1, 32, 1, 64, 64, generator=g),
        },
        path,
    )


def _worker(*args):
    return [sys.executable, WORKER, *args]


def _check(name, baseline, expected, actual):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    result = assert_close_three_way(baseline, expected, actual, name=name)
    print(f"{name}: {result}")


def _official(common, dtypes, out_dir):
    """Start the reference runs, one per dtype, alongside whatever runs on the device."""
    return {
        dtype: start(_worker(*common, "--device", "official", "--dtype", dtype,
                             "--output", str(out_dir / f"official_{dtype}.pt")), out_dir / f"official_{dtype}.log")
        for dtype in dtypes
    }


def test_backbone_layer_and_unet(require_neuron_device, tmp_path):
    inputs = tmp_path / "inputs.pt"
    _inputs(inputs)
    common = ["--mode", "backbone", "--model-path", require_neuron_device, "--inputs", str(inputs)]
    references = _official(common, ("float32", "bfloat16"), tmp_path)  # host only, meanwhile
    run(["torchrun", "--nproc-per-node", str(TP), WORKER, *common, "--device", "neuron",
         "--output", str(tmp_path / "neuron.pt")], tmp_path / "neuron.log")
    for proc in references.values():
        wait(proc)
    device = torch.load(tmp_path / "neuron.pt")
    cpu32, cpu16 = torch.load(tmp_path / "official_float32.pt"), torch.load(tmp_path / "official_bfloat16.pt")
    # Rank 0's prompt K/V is its KV head (TP16: 2 query heads per rank share one of 8).
    head = slice(device["tp_rank"] * 8 // TP, device["tp_rank"] * 8 // TP + 1)
    for key in ("key", "value"):
        _check(f"layer 0 prefill {key}", cpu32[key][:, :PROMPT, head], cpu16[key][:, :PROMPT, head],
               device[key][:, :PROMPT])
    _check("layer 0 denoise step", cpu32["denoise"], cpu16["denoise"], device["denoise"])
    _check("patch embed", cpu32["patch_embed"], cpu16["patch_embed"], device["patch_embed"])
    _check("final layer", cpu32["final_layer"], cpu16["final_layer"], device["final_layer"])


@pytest.fixture(scope="module")
def vae_references(require_neuron_device, tmp_path_factory):
    """The official decodes (float32, float16) of the shared inputs, once for both splits."""
    out = tmp_path_factory.mktemp("vae_official")
    _inputs(out / "inputs.pt")
    common = ["--mode", "vae", "--model-path", require_neuron_device, "--inputs", str(out / "inputs.pt")]
    for proc in _official(common, ("float32", "float16"), out).values():
        wait(proc)
    return {dtype: torch.load(out / f"official_{dtype}.pt")["image"] for dtype in ("float32", "float16")}


@pytest.mark.parametrize("vae_parallel", [2, 1])
def test_vae_decoder(require_neuron_device, vae_references, tmp_path, vae_parallel):
    inputs = tmp_path / "inputs.pt"
    _inputs(inputs)
    args = ["--mode", "vae", "--model-path", require_neuron_device, "--inputs", str(inputs), "--device", "neuron",
            "--dtype", "float16", "--output", str(tmp_path / "neuron.pt")]
    if vae_parallel > 1:
        run(["torchrun", "--nproc-per-node", str(WORLD), WORKER, *args, "--vae-parallel", str(vae_parallel)],
            tmp_path / "neuron.log")
    else:
        run(_worker(*args), tmp_path / "neuron.log", env={"NEURON_RT_VISIBLE_CORES": "0"})
    _check(f"VAE decode (vae_parallel {vae_parallel})", vae_references["float32"], vae_references["float16"],
           torch.load(tmp_path / "neuron.pt")["image"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
