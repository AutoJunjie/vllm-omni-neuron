# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the HunyuanImage-3.0 hardware tests: subprocess runs, SSIM, traces."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(REPO, "examples", "hunyuan_image3")
MARS_PROMPT = (
    "A cinematic photo of a glass observatory on Mars at sunrise, volumetric light, ultra detailed"
)


def start(cmd: list[str], log_path, env: dict | None = None) -> subprocess.Popen:
    """Start ``cmd`` in its own process from the repo root, output to ``log_path``.

    ``GLOO_SOCKET_IFNAME=lo`` keeps the host collectives off the node's hostname, and
    ``NEURON_RT_VISIBLE_CORES`` is dropped unless ``env`` sets it (the engine, or
    torchrun's workers, assign the cores).
    """
    full_env = dict(os.environ)
    full_env.pop("NEURON_RT_VISIBLE_CORES", None)
    full_env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    full_env.update(env or {})
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=REPO, env=full_env, stdout=log, stderr=subprocess.STDOUT)
    proc.log, proc.log_path = log, str(log_path)
    return proc


def wait(proc: subprocess.Popen, timeout=None) -> str:
    """Wait for a :func:`start`-ed process; its log text, or an error with the log's tail."""
    returncode = proc.wait(timeout=timeout)
    proc.log.close()
    text = open(proc.log_path, errors="replace").read()
    if returncode != 0:
        tail = "\n".join(text.splitlines()[-40:])
        raise AssertionError(f"{' '.join(map(str, proc.args[:4]))} ... exited {returncode}; {proc.log_path}:\n{tail}")
    return text


def run(cmd: list[str], log_path, env: dict | None = None, timeout=None) -> str:
    return wait(start(cmd, log_path, env), timeout)


def run_example(script: str, args: list[str], log_path, env: dict | None = None, timeout=None) -> str:
    """Run ``examples/hunyuan_image3/<script>`` in its own process; return its log text.

    Each run gets a fresh process: the engine owns every NeuronCore of the stage while it
    lives.
    """
    return run([sys.executable, os.path.join(EXAMPLES, script), *args], log_path, env, timeout)


def profile_times(log_text: str) -> list[float]:
    """The ``[profile] run i/n: X s`` request times ``run.py --profile`` prints."""
    return [float(t) for t in re.findall(r"\[profile\] run \d+/\d+: ([0-9.]+)s", log_text)]


def load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the RGB channels (Wang et al. 2004: 11 x 11 Gaussian, sigma 1.5)."""
    from scipy.ndimage import gaussian_filter

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    # truncate=3.5 makes the Gaussian window 11 x 11 at sigma 1.5, as in the paper.
    blur = lambda x: gaussian_filter(x, sigma=1.5, truncate=3.5)  # noqa: E731
    scores = []
    for channel in range(a.shape[-1]):
        x, y = a[..., channel], b[..., channel]
        mx, my = blur(x), blur(y)
        sxx = blur(x * x) - mx * mx
        syy = blur(y * y) - my * my
        sxy = blur(x * y) - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sxx + syy + c2)
        scores.append(float((num / den).mean()))
    return float(np.mean(scores))
