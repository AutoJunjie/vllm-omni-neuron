# SPDX-License-Identifier: Apache-2.0
"""The official HunyuanImage-3.0 implementation on the host, as the tests' reference.

The checkpoint ships Tencent's own modeling code (``modeling_hunyuan_image_3.py`` and
its siblings, loaded with ``trust_remote_code``); the accuracy tests run it on the CPU
in float32 (baseline) and bfloat16 (expected), so the device is measured against an
implementation that shares no code with the plugin or with vLLM-Omni's port.

The official code assumes CUDA in a few places; :func:`official_module` sends those
tensors to the host. Modules are built under ``accelerate.init_empty_weights`` and only
the parts a test needs are materialised and loaded, in the official dtype rule: every
parameter in the requested dtype except those the code declares float32 (the MoE gate),
as ``from_pretrained(torch_dtype=...)`` loads them.
"""

from __future__ import annotations

import json
import os
import sys

import torch


def official_module(model_path: str):
    """The checkpoint's ``modeling_hunyuan_image_3`` module, patched to run on the host."""
    for name in ("empty", "zeros", "ones", "tensor", "arange", "full"):
        original = getattr(torch, name)
        if getattr(original, "_hy3_host", False):
            continue

        def host(*args, _original=original, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        host._hy3_host = True
        setattr(torch, name, host)
    torch.cuda.set_device = lambda *args, **kwargs: None  # HunyuanMoE.forward calls it
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    cls = get_class_from_dynamic_module(
        "modeling_hunyuan_image_3.HunyuanImage3ForCausalMM", model_path
    )
    return sys.modules[cls.__module__], config


def build(factory, dtype):
    """``factory()`` under ``init_empty_weights``, with ``dtype`` as the default dtype."""
    from accelerate import init_empty_weights

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights(include_buffers=False):
            module = factory()
    finally:
        torch.set_default_dtype(previous)
    return module


def shards(model_path: str, prefix: str, names=None):
    """The checkpoint's ``<prefix>*`` tensors, one ``{name without prefix: tensor}`` per file.

    ``names``: only these (prefix stripped), when given.
    """
    from safetensors import safe_open

    with open(os.path.join(model_path, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    wanted = {
        key: file
        for key, file in weight_map.items()
        if key.startswith(prefix) and (names is None or key[len(prefix) :] in names)
    }
    for filename in sorted(set(wanted.values())):
        with safe_open(os.path.join(model_path, filename), framework="pt", device="cpu") as f:
            yield {key[len(prefix) :]: f.get_tensor(key) for key, file in wanted.items() if file == filename}


def read(model_path: str, prefix: str) -> dict:
    """The checkpoint's ``<prefix>*`` tensors as one state dict, the prefix stripped."""
    return {name: tensor for shard in shards(model_path, prefix) for name, tensor in shard.items()}


def load(module, model_path: str, prefix: str, strict: bool = True):
    """Materialise ``module`` on the host and load the checkpoint's ``<prefix>*`` tensors.

    One checkpoint file at a time, so the whole model peaks at its own size plus a file.
    """
    module.to_empty(device="cpu")
    expected, loaded = set(module.state_dict()), set()
    for shard in shards(model_path, prefix, expected if not strict else None):
        module.load_state_dict(shard, strict=False)
        loaded |= set(shard)
    missing, unexpected = sorted(expected - loaded), sorted(loaded - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(f"{prefix}: missing {missing[:6]}, unexpected {unexpected[:6]}")
    return module.eval().requires_grad_(False)
