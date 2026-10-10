# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of the HunyuanImage-3.0 text-to-image pipeline.

Everything here is import-light on purpose. ``vllm_omni_neuron._register_pipelines``
imports this package while vLLM-Omni is still initialising its own plugins, so the
pipeline class and the upstream modules it needs are resolved lazily through PEP 562
``__getattr__`` — by which point ``vllm_omni`` is fully imported.
"""

from typing import Any

PIPELINE_REGISTRY = [
    {
        "model_arch": "HunyuanImage3ForCausalMM",
        "class_name": "NeuronHunyuanImage3Pipeline",
        "pre_process_func_name": "get_hunyuan_image_3_pre_process_func",
    },
]

__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronHunyuanImage3Pipeline",
    "get_hunyuan_image_3_pre_process_func",
]


def get_hunyuan_image_3_pre_process_func(od_config: Any):
    """Upstream's request pre-processor — the Neuron path needs no changes to it."""
    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        get_hunyuan_image_3_pre_process_func as _upstream,
    )

    return _upstream(od_config)


def __getattr__(name: str) -> Any:
    if name == "NeuronHunyuanImage3Pipeline":
        from .pipeline_hunyuan_image3 import NeuronHunyuanImage3Pipeline

        return NeuronHunyuanImage3Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
