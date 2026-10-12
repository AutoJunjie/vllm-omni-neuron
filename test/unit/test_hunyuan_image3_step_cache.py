# SPDX-License-Identifier: Apache-2.0
"""``_StepCache`` must take the same decisions as Cache-DiT's DBCache.

The rules are few but each one silently changes the image when wrong: caching during
warmup, caching a fourth step in a row, or treating a NaN diff (the first step's
division by a zero residual) as similar.

    VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_step_cache.py
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest  # noqa: E402


def _cache(**overrides):
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import _StepCache

    return _StepCache(**{**_StepCache.DEFAULTS, **overrides})


def _run(cache, diffs):
    decisions = []
    for index, diff in enumerate(diffs):
        use = cache.wants_cache(index, diff)
        cache.record(index, diff, use)
        if not use:
            cache.update(object(), object())
        decisions.append(use)
    return decisions


def test_parse() -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import _StepCache

    assert _StepCache.parse(None) is None and _StepCache.parse(False) is None
    assert _StepCache.parse(True) == _StepCache.DEFAULTS
    assert _StepCache.parse({"enabled": False}) is None
    assert _StepCache.parse({"residual_diff_threshold": 0.1})["residual_diff_threshold"] == 0.1
    with pytest.raises(ValueError):
        _StepCache.parse({"rdt": 0.1})


def test_warmup_continuous_and_threshold() -> None:
    # Every step similar: 4 warmup computes, then 3 cached, 1 forced compute, repeat.
    assert _run(_cache(), [0.0] * 12) == [False] * 4 + [True] * 3 + [False] + [True] * 3 + [False]
    # Above the threshold nothing caches; NaN (first step) never does.
    assert not any(_run(_cache(), [float("nan")] + [0.5] * 9))
    # A compute in between restarts the run of cached steps.
    assert _run(_cache(), [0.0] * 4 + [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]) == (
        [False] * 4 + [True, True, False, True, True, True, False]
    )


def test_max_cached_steps() -> None:
    cache = _cache(max_cached_steps=2, max_continuous_cached_steps=-1)
    assert _run(cache, [0.0] * 10).count(True) == 2


if __name__ == "__main__":
    test_parse()
    test_warmup_continuous_and_threshold()
    test_max_cached_steps()
    print("step cache rules OK")
