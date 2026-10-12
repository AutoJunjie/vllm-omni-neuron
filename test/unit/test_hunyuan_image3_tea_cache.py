# SPDX-License-Identifier: Apache-2.0
"""``_TeaCache`` must skip exactly the steps vllm-omni's GPU pipeline skips.

The decision is a small accumulator over the timestep embedding, and it is easy to drift
from upstream by one step -- accumulating before or after the comparison, resetting on
the wrong branch, or computing the first step's distance. This replays upstream's loop
(``HunyuanImage3Pipeline`` denoise loop in vllm-omni's ``hunyuan_image3_transformer.py``)
as a reference over synthetic embeddings and checks the plan matches, then checks the
configuration comes from vllm-omni's ``TeaCacheConfig``.

    VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_tea_cache.py
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402


class _Embed(torch.nn.Module):
    """time_embed stand-in: a fixed random direction scaled by a function of t."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.base = torch.randn(1, 64)
        self.wiggle = torch.randn(1, 64)

    def forward(self, t):
        t = t.float().reshape(-1, 1)
        return self.base * (1.0 + t / 1000.0) + self.wiggle * torch.sin(t / 37.0)


def _upstream_plan(embed, timesteps, threshold, coefficients):
    rescale = np.poly1d(coefficients)
    acc, prev, cnt, plan = 0.0, None, 0, []
    for t in timesteps:
        should_compute = True
        with torch.no_grad():
            cur = embed(t.unsqueeze(0))
        if cnt > 0 and prev is not None:
            rel = ((cur - prev).abs().mean() / (prev.abs().mean() + 1e-8)).cpu().item()
            acc += abs(float(rescale(rel)))
            if acc < threshold:
                should_compute = False
            else:
                acc = 0.0
        prev = cur.detach()
        plan.append(should_compute)
        cnt += 1
    return plan


def test_plan_matches_upstream_loop() -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import _TeaCache

    embed = _Embed()
    timesteps = torch.linspace(1000.0, 10.0, 50)
    for threshold in (0.05, 0.2, 0.5):
        config = _TeaCache.parse({"rel_l1_thresh": threshold})
        cache = _TeaCache(**config)
        got = cache.plan(embed, timesteps)
        want = _upstream_plan(embed, timesteps, threshold, config["coefficients"])
        assert got == want, f"threshold {threshold}: {got} != {want}"
        assert got[0], "the first step always computes"


def test_config_comes_from_vllm_omni() -> None:
    from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import _TeaCache

    upstream = TeaCacheConfig(transformer_type="HunyuanImage3Pipeline")
    assert _TeaCache.parse(True) == {
        "rel_l1_thresh": upstream.rel_l1_thresh,
        "coefficients": list(upstream.coefficients),
    }
    assert _TeaCache.parse(None) is None and _TeaCache.parse({"enabled": False}) is None
    with pytest.raises(ValueError):
        _TeaCache.parse({"thresh": 0.2})


if __name__ == "__main__":
    test_plan_matches_upstream_loop()
    test_config_comes_from_vllm_omni()
    print("TeaCache plan matches upstream")
