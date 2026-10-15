# SPDX-License-Identifier: Apache-2.0
"""CPU checks of HunyuanImage3 request and configuration handling.

Each test drives one pipeline / model method on a stand-in ``self``, so none needs the
checkpoint:

* the MoE router keeps exactly ``top_k`` experts when the k-th probability ties;
* prompt buckets: ``prefill_len`` is always one, a prompt takes the smallest that fits;
* ``vae_parallel`` (the row split of the device decode) is unused with the host decode;
* ``od_config.revision`` reaches the hub snapshot;
* the request pre-processor reads the checkpoint at ``od_config.revision`` too;
* ``compare_step_traces.py`` handles traces without CFG and, with a threshold, fails
  on traces that are not comparable (timestep, missing steps, non-finite values);
* ``geneval_generate.py`` refuses to resume into a folder generated otherwise;
* the ``cfg_combine: mixin`` dispatch builds the plain pipeline unless asked.

    VLLM_NEURON_CPU_MODE=1 python -m pytest test/unit/test_hunyuan_image3_request_options.py
"""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def _pipeline_stub(**model_config):
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        NeuronHunyuanImage3Pipeline,
        _prompt_buckets,
    )

    stub = SimpleNamespace(
        od_config=SimpleNamespace(model_config=model_config, model="org/model", revision="abc123"),
        _prefill_buckets=_prompt_buckets(model_config),
        vae_device=model_config.get("vae_device", "neuron"),
    )
    return NeuronHunyuanImage3Pipeline, stub


def test_prefill_len_is_always_a_bucket() -> None:
    cls, stub = _pipeline_stub(prefill_len=16384, prefill_buckets=[512, 6144, 8192])
    assert cls._prefill_bucket(stub, 300) == 512
    assert cls._prefill_bucket(stub, 9000) == 16384
    with pytest.raises(ValueError, match="largest prompt bucket is 16384"):
        cls._prefill_bucket(stub, 20000)


def test_cfg_mixin_dispatch_builds_either_class() -> None:
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.cfg_mixin_pipeline import install

    class Pipeline:
        def __init__(self, od_config, prefix=""):
            self.od_config = od_config

    install(Pipeline)
    plain = Pipeline(SimpleNamespace(model_config={}))
    assert type(plain) is Pipeline
    with mock.patch(
        "vllm_omni_neuron.diffusion.models.hunyuan_image3.cfg_mixin_pipeline.cfg_mixin_pipeline_class",
        return_value=type("Mixin", (Pipeline,), {}),
    ):
        assert type(Pipeline.__new__(Pipeline, SimpleNamespace(model_config={"cfg_combine": "mixin"}))).__name__ == "Mixin"


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


def _trace(path, pred_raw, pred, offset=0.0, steps=(0,), timestep=999.0):
    os.makedirs(path, exist_ok=True)
    latents = torch.ones(1, 4, 8, 8)
    for step in steps:
        torch.save(
            {"timestep": torch.tensor(timestep - step), "latents_in": latents, "pred_raw": pred_raw + offset,
             "pred": pred + offset, "latents_out": latents - 0.1 * (pred + offset)},
            os.path.join(path, f"step_{step:02d}.pt"),
        )


@pytest.mark.parametrize("cfg", [1, 2])
def test_compare_step_traces_checks_against_threshold(tmp_path, cfg) -> None:
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(cfg, 4, 8, 8, generator=g)
    pred = raw[:1] if cfg == 1 else raw[1:] + 2.5 * (raw[:1] - raw[1:])
    _trace(tmp_path / "ref", raw, pred)
    _trace(tmp_path / "close", raw, pred, offset=1e-4)
    _trace(tmp_path / "far", raw, pred, offset=1.0)
    script = os.path.join(REPO, "examples", "hunyuan_image3", "compare_step_traces.py")

    def run(name, *extra):
        return subprocess.run(
            [sys.executable, script, "--reference", f"ref={tmp_path / 'ref'}",
             f"{name}={tmp_path / name}", "--max-rel-l2", "1e-2", *extra],
            capture_output=True, text=True,
        )

    close = run("close")
    assert close.returncode == 0, close.stdout + close.stderr
    assert ("cond mean" in close.stdout) == (cfg == 2)
    assert run("far").returncode == 1
    # Not comparable: another timestep, a missing step, a non-finite prediction.
    _trace(tmp_path / "timestep", raw, pred, timestep=100.0)
    _trace(tmp_path / "ref2", raw, pred, steps=(0, 1))
    _trace(tmp_path / "nan", raw, pred, offset=float("nan"))
    assert run("timestep").returncode == 1
    assert run("nan").returncode == 1
    short = subprocess.run(
        [sys.executable, script, "--reference", f"ref={tmp_path / 'ref2'}", f"close={tmp_path / 'close'}",
         "--max-rel-l2", "1e-2"], capture_output=True, text=True)
    assert short.returncode == 1 and "step 1 missing" in short.stdout
    subset = subprocess.run(
        [sys.executable, script, "--reference", f"ref={tmp_path / 'ref2'}", f"close={tmp_path / 'close'}",
         "--max-rel-l2", "1e-2", "--steps", "0"], capture_output=True, text=True)
    assert subset.returncode == 0, subset.stdout + subset.stderr


def test_geneval_resume_refuses_a_changed_prompt_set(tmp_path) -> None:
    script = os.path.join(REPO, "examples", "hunyuan_image3", "geneval_generate.py")

    def run(prompt, seed=0):
        metadata = tmp_path / "metadata.jsonl"
        metadata.write_text(f'{{"tag": "single_object", "prompt": "{prompt}"}}\n')
        # No server: every image is already on disk, so only the record check runs.
        return subprocess.run(
            [sys.executable, script, "--metadata", str(metadata), "--outdir", str(tmp_path / "out"),
             "--samples", "1", "--seed", str(seed), "--url", "http://127.0.0.1:9/none"],
            capture_output=True, text=True,
        )

    os.makedirs(tmp_path / "out" / "00000" / "samples")
    (tmp_path / "out" / "00000" / "samples" / "0000.png").write_bytes(b"png")
    assert run("a photo of a cat").returncode != 0  # images without a generation record
    (tmp_path / "out" / "00000" / "generation.json").write_text(
        '{"metadata": {"tag": "single_object", "prompt": "a photo of a cat"}, "seed": 0, "size": 1024}')
    assert run("a photo of a cat").returncode == 0  # resumes: nothing to generate
    changed = run("a photo of a dog")
    assert changed.returncode != 0 and "use a new --outdir" in changed.stderr
    assert run("a photo of a cat", seed=1).returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
