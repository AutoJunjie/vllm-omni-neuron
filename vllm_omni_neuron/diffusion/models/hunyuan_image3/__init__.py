# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of the HunyuanImage-3.0 text-to-image pipeline.

Everything here is import-light on purpose. ``vllm_omni_neuron._register_pipelines``
imports this package while vLLM-Omni is still initialising its own plugins, so the
pipeline class and the upstream modules it needs are resolved lazily through PEP 562
``__getattr__`` — by which point ``vllm_omni`` is fully imported.
"""

import os
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


def resolve_snapshot(model: str, revision: str | None = None) -> str:
    """A local directory for ``model``: as given, or its hub snapshot at ``revision``.

    The pipeline (config, generation config, tokenizer, weights) and the request
    pre-processor both read the checkpoint through this, so they see the same revision.
    """
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision)


def get_hunyuan_image_3_pre_process_func(od_config: Any):
    """Upstream's request pre-processor, reading the config of ``od_config.revision``.

    Upstream's factory reads ``od_config.model`` without a revision, so it gets the
    resolved snapshot directory instead.
    """
    import copy

    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        get_hunyuan_image_3_pre_process_func as _upstream,
    )

    resolved = copy.copy(od_config)
    resolved.model = resolve_snapshot(od_config.model, getattr(od_config, "revision", None))
    return _upstream(resolved)


def __getattr__(name: str) -> Any:
    if name == "NeuronHunyuanImage3Pipeline":
        from .cfg_mixin_pipeline import install
        from .pipeline_hunyuan_image3 import NeuronHunyuanImage3Pipeline

        # model_config.cfg_combine: mixin builds the NeuronCFGParallelMixin subclass instead.
        install(NeuronHunyuanImage3Pipeline)
        return NeuronHunyuanImage3Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
