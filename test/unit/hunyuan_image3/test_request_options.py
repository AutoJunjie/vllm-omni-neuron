# SPDX-License-Identifier: Apache-2.0
"""CPU checks of HunyuanImage3 request and configuration handling.

Each test drives one pipeline / model method on a stand-in ``self``, so none needs the
checkpoint:

* the MoE router keeps exactly ``top_k`` experts when the k-th probability ties;
* a request may ask for text-to-image or think / recaption (with or without
  conditioning images) at 1024 x 1024, and with CFG replicas it must use CFG; anything
  else raises before any device work;
* the request's ``bot_task`` falls back to the stage's ``model_config.bot_task``;
* model_config keys of features this build does not include raise when set, while the
  prompt-bucket, think / recaption and image-editing keys pass;
* prompt buckets: ``prefill_len`` is always one, a prompt takes the smallest that fits;
* the conditioning images encode on device on one rank, or on the host;
* ``vae_parallel`` (the row split of the device decode) is unused with the host decode;
* ``od_config.revision`` reaches the hub snapshot;
* the request pre-processor reads the checkpoint at ``od_config.revision`` too.

    VLLM_NEURON_CPU_MODE=1 python -m pytest test/unit/hunyuan_image3/test_request_options.py
"""

import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest  # noqa: E402
import torch  # noqa: E402


def test_router_keeps_exactly_top_k_on_a_tie() -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        NeuronHunyuanMoE,
    )

    experts, top_k = 64, 8
    # Logits whose 8th and 9th..64th probabilities tie: a ">= k-th" threshold admits all.
    logits = torch.zeros(3, experts)
    logits[:, :7] = 2.0
    stub = SimpleNamespace(gate_weight=torch.eye(experts), top_k=top_k, norm_topk_prob=True)
    dense = NeuronHunyuanMoE._routing_dense(stub, logits)
    assert ((dense > 0).sum(dim=-1) == top_k).all()
    assert torch.allclose(dense.sum(dim=-1), torch.ones(3))
    # Same weights as upstream's topk -> renormalise.
    values, chosen = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    expected = torch.zeros_like(dense).scatter(-1, chosen, values / values.sum(-1, keepdim=True))
    assert torch.allclose(dense, expected)


def _pipeline_stub(cfg_parallel: bool = True, **model_config):
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        NeuronHunyuanImage3Pipeline,
        _prompt_buckets,
    )

    stub = SimpleNamespace(
        od_config=SimpleNamespace(model_config=model_config, model="org/model", revision="abc123"),
        _prefill_buckets=_prompt_buckets(model_config),
        vae_device=model_config.get("vae_device", "neuron"),
        cfg_parallel=cfg_parallel,
    )
    return NeuronHunyuanImage3Pipeline, stub


_SUPPORTED = dict(bot_task="image", height=1024, width=1024, guidance=2.5)


@pytest.mark.parametrize(
    "bot_task", [None, "auto", "image", "None", "think", "recaption", "think_recaption"]
)
def test_supported_requests_pass(bot_task) -> None:
    cls, stub = _pipeline_stub()
    cls._check_request(stub, **{**_SUPPORTED, "bot_task": bot_task})


def test_ar_tasks_match_the_generator() -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_ar import AR_BOT_TASKS
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        SUPPORTED_BOT_TASKS,
    )

    assert set(AR_BOT_TASKS) <= set(SUPPORTED_BOT_TASKS)


@pytest.mark.parametrize(
    "change, error, match",
    [
        ({"bot_task": "img_ratio"}, NotImplementedError, "bot_task 'img_ratio'"),
        ({"height": 512, "width": 512}, ValueError, "512x512"),
        ({"height": 1024, "width": 768}, ValueError, "1024x768"),
        ({"guidance": 1.0}, ValueError, "turns CFG off"),
    ],
)
def test_unsupported_requests_raise(change, error, match) -> None:
    cls, stub = _pipeline_stub()
    with pytest.raises(error, match=match):
        cls._check_request(stub, **{**_SUPPORTED, **change})


def test_no_cfg_runs_without_cfg_replicas() -> None:
    cls, stub = _pipeline_stub(cfg_parallel=False)
    cls._check_request(stub, **{**_SUPPORTED, "guidance": 1.0})


def test_bot_task_falls_back_to_the_stage_config() -> None:
    cls, stub = _pipeline_stub(bot_task="image")
    text = SimpleNamespace(prompts=["a red teapot"])
    assert cls._request_bot_task(stub, text, {}) == "image"
    asked = SimpleNamespace(prompts=[{"prompt": "a red teapot", "bot_task": "think"}])
    assert cls._request_bot_task(stub, asked, {}) == "think"
    assert cls._request_bot_task(stub, text, {"bot_task": "recaption"}) == "recaption"


@pytest.mark.parametrize(
    "model_config",
    [
        {"step_cache": {"residual_diff_threshold": 0.12}},
        {"tea_cache": True},
        {"cfg_combine": "mixin"},
        {"tp_sequence_parallel": True},
        {"vae_device_core": 16},
    ],
)
def test_later_features_in_model_config_raise(model_config) -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        _check_model_config,
    )

    with pytest.raises(NotImplementedError, match=next(iter(model_config))):
        _check_model_config(model_config)


def test_disabled_later_features_are_accepted() -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        _check_model_config,
    )

    _check_model_config({"step_cache": False, "tea_cache": False, "bot_task": "image"})


def test_m2_keys_are_accepted() -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        _check_model_config,
    )

    _check_model_config(
        {
            "prefill_len": 2560,
            "prefill_buckets": [2560, 8192],
            "ar_max_new_tokens": 1024,
            "bot_task": "think",
            "vae_encode_device": "neuron",
            "vae_encode_rank": 2,
        }
    )


def test_prefill_len_is_always_a_bucket() -> None:
    cls, stub = _pipeline_stub(prefill_len=16384, prefill_buckets=[512, 6144, 8192])
    assert cls._prefill_bucket(stub, 300) == 512
    assert cls._prefill_bucket(stub, 9000) == 16384
    with pytest.raises(ValueError, match="largest prompt bucket is 16384"):
        cls._prefill_bucket(stub, 20000)


def test_single_bucket_without_prefill_buckets() -> None:
    cls, stub = _pipeline_stub(prefill_len=512)
    assert cls._prefill_bucket(stub, 512) == 512
    with pytest.raises(ValueError, match="largest prompt bucket is 512"):
        cls._prefill_bucket(stub, 513)


@pytest.mark.parametrize(
    "model_config, expected",
    [
        ({}, 0),  # one rank, no row split: the encode runs on rank 0's core
        ({"vae_encode_device": "cpu"}, None),
        ({"vae_device": "cpu"}, None),  # no device VAE: the host encodes
        ({"vae_encode_rank": 3}, None),  # beyond the world: the host encodes
    ],
)
def test_vae_encode_rank(model_config, expected) -> None:
    cls, stub = _pipeline_stub(**model_config)
    stub._vae_split = lambda: (1, None)
    assert cls._vae_encode_rank(stub) == expected


def test_vae_encode_device_is_checked() -> None:
    cls, stub = _pipeline_stub(vae_encode_device="gpu")
    stub._vae_split = lambda: (1, None)
    with pytest.raises(ValueError, match="vae_encode_device"):
        cls._vae_encode_rank(stub)


def test_host_decode_ignores_the_row_split() -> None:
    cls, stub = _pipeline_stub(vae_device="cpu", vae_parallel=2)
    assert cls._vae_split(stub) == (1, None)


def test_revision_reaches_the_snapshot() -> None:
    cls, stub = _pipeline_stub()
    with mock.patch("huggingface_hub.snapshot_download", return_value="/snapshots/abc123") as download:
        assert cls._resolve_model_path(stub) == "/snapshots/abc123"
        assert cls._resolve_model_path(stub) == "/snapshots/abc123"  # resolved once
    download.assert_called_once_with("org/model", revision="abc123")


def test_pre_processor_reads_the_same_revision() -> None:
    from vllm_omni_neuron.diffusion.models import hunyuan_image3

    od_config = SimpleNamespace(model="org/model", revision="abc123")
    upstream = "vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3.get_hunyuan_image_3_pre_process_func"
    with mock.patch("huggingface_hub.snapshot_download", return_value="/snapshots/abc123") as download, \
            mock.patch(upstream, side_effect=lambda config: config.model) as factory:
        assert hunyuan_image3.get_hunyuan_image_3_pre_process_func(od_config) == "/snapshots/abc123"
    download.assert_called_once_with("org/model", revision="abc123")
    factory.assert_called_once()
    assert od_config.model == "org/model"  # the caller's config is left as it was


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
