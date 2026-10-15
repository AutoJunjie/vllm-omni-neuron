# SPDX-License-Identifier: Apache-2.0
"""Tier 2: one denoising step of the whole pipeline against the official code, three ways.

The device runs the default stage config (TP16 x two CFG replicas, NKI kernels, the
UNet split) and records its step trace: the request's templated inputs, then per step
the input latents, the raw CFG-batch prediction, the guided prediction and the
scheduler's updated latents. The official implementation then replays those steps on
the host (``hy3_official_step.py``: the checkpoint's own modeling code, scheduler and
CFG operator) from the same input latents, once in float32 (baseline) and once in
bfloat16 (expected). ``assert_close_three_way`` passes when the device's error
distribution matches the reference's bfloat16 rounding error: what is left is what the
port, the kernels and the device add.

Steps: ``HY3_ACCURACY_STEPS`` (default ``0``, the first denoising step -- its latents
are the seed's noise, so no forcing is involved). Each reference step takes tens of
minutes and the float32 model ~320 GB of host memory.

    pytest -s test/neuron/test_hunyuan_image3_pipeline_accuracy.py
"""

import os
import sys

import pytest
import torch
from hy3_helpers import MARS_PROMPT, run_example, start, wait

STEPS = os.environ.get("HY3_ACCURACY_STEPS", "0")
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy3_official_step.py")


def test_denoise_steps_three_way(require_neuron_device, tmp_path):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    model_path = require_neuron_device
    device_dir, logs = tmp_path / "device", tmp_path
    run_example(
        "run.py",
        ["--model-path", model_path, "--seed", "42", "--prompt", MARS_PROMPT,
         "--output", str(tmp_path / "device.png")],
        str(logs / "device.log"),
        # Record only the steps the reference replays (it also reads step 0's inputs).
        env={"HY3_TRACE_DIR": str(device_dir), "HY3_TRACE_STEPS": ",".join(sorted({"0", *STEPS.split(",")}))},
    )
    # Both references at once, each on half the host's cores (~320 + 160 GB of memory).
    traces = {dtype: tmp_path / f"official_{dtype}" for dtype in ("float32", "bfloat16")}
    threads = str(max(1, (os.cpu_count() or 2) // 2))
    references = [
        start([sys.executable, REFERENCE, "--model-path", model_path, "--trace", str(device_dir),
               "--steps", STEPS, "--dtype", dtype, "--threads", threads, "--output", str(out)],
              logs / f"official_{dtype}.log")
        for dtype, out in traces.items()
    ]
    for proc in references:
        wait(proc)
    for step in (int(s) for s in STEPS.split(",")):
        name = f"step_{step:02d}.pt"
        device = torch.load(device_dir / name)
        baseline = torch.load(traces["float32"] / name)
        expected = torch.load(traces["bfloat16"] / name)
        for reference in (baseline, expected):
            assert torch.equal(device["latents_in"], reference["latents_in"]), "the step inputs differ"
        # The raw CFG batch (cond, uncond), the guided prediction the scheduler takes and
        # the scheduler's updated latents.
        for key in ("pred_raw", "pred", "latents_out"):
            result = assert_close_three_way(
                baseline[key].float(), expected[key].float(), device[key].float(),
                name=f"step {step} {key}",
            )
            print(f"step {step} {key}: {result}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
