# SPDX-License-Identifier: Apache-2.0
"""Tier 3: full generations against reviewed golden outputs (SSIM), and their latency.

A regression check against a previous Neuron output with identical inputs and
configuration, not a comparison with the reference implementation: a per-step error too
small for tier 2 can still compound over 50 steps into a washed-out image, which only a
full run shows.

The goldens live in ``HY3_GOLDEN_DIR`` (default ``test/neuron/goldens``) with
``manifest.json``, which binds each case's golden to how it was produced:

- the request (``run.py`` arguments), the stage config (a hash of its parsed content),
  the checkpoint (revision, and a fingerprint of its model config, generation config --
  guidance, flow shift, steps -- and weight index) and the
  environment (package versions), the plugin commit;
- the request's token ids and initial latents, read from a one-step trace of the run
  (``HY3_TRACE_STEPS=0``); the latents are archived next to the goldens;
- the golden's sha256 and the warm request latency.

A regression run checks all of these before it compares images: a missing or altered
golden, a different checkpoint, stage config, request or initial latents fails (package
versions that differ are reported). To produce the goldens, review a run's outputs and
rerun with ``HY3_UPDATE_GOLDEN=1`` and ``HY3_MODEL_REVISION`` set to the checkpoint's
revision (``HY3_PLUGIN_COMMIT`` when the checkout is not a git tree). The cases run in the
listed order, so the edit case's input image (a copy of the text-to-image golden, kept as
its own file) exists by the time it runs.

Latency: the faster of each case's two warm requests must stay within
``HY3_LATENCY_TOLERANCE`` (default 10%; 25% for the edit case, whose host-side SigLIP2
encode varies by seconds) of the manifest's, the slower of the reviewed run's two.

    pytest -s test/neuron/test_hunyuan_image3_e2e_accuracy.py
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess

import pytest
import yaml
from hy3_helpers import (
    EXAMPLES,
    MARS_PROMPT,
    REPO,
    load_rgb,
    profile_times,
    psnr,
    run_example,
    ssim,
)

GOLDEN_DIR = os.environ.get("HY3_GOLDEN_DIR", os.path.join(os.path.dirname(__file__), "goldens"))
MANIFEST = os.path.join(GOLDEN_DIR, "manifest.json")
UPDATE = os.environ.get("HY3_UPDATE_GOLDEN") == "1"
# SSIM floor against the golden. Measured on the Mars text-to-image output: a
# byte-identical rerun scores 1.0; sound changes of parallel layout or kernel keep the
# trajectory and score 0.967-0.989 (TP32 vs TP16 0.975 at 34.8 dB, the tile-parallel VAE
# 0.989 at 42.0 dB, tp_sequence_parallel 0.967 at 32.7 dB); switching the step cache off
# changes the trajectory, 0.946 at 30.6 dB; an unrelated image scores 0.18.
SSIM_MIN = float(os.environ.get("HY3_SSIM_MIN", "0.95"))
LATENCY_TOLERANCE = float(os.environ.get("HY3_LATENCY_TOLERANCE", "0.10"))
# The edit case encodes its conditioning image with SigLIP2 on the host, whose time
# varies: warm requests measured 16.0-21.0 s on one node and build.
CASE_LATENCY_TOLERANCE = {"edit": 0.25}
EDIT_INPUT = "hunyuan_image3_edit_input.png"


def _tile_vae_stage(tmp_path) -> str:
    """The default stage config with vLLM-Omni's tile-parallel VAE instead of the row split."""
    with open(os.path.join(EXAMPLES, "hunyuan_image3_stage.yaml")) as handle:
        config = yaml.safe_load(handle)
    engine_args = config["stage_args"][0]["engine_args"]
    engine_args["model_config"]["vae_parallel"] = 1
    engine_args["parallel_config"]["vae_patch_parallel_size"] = 16
    engine_args["vae_use_tiling"] = True
    path = tmp_path / "hunyuan_image3_stage_vae_tiles.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


# name -> (golden file, run.py arguments), in run order. The edit case's input image is
# EDIT_INPUT; the tile-VAE case also gets its stage config (_tile_vae_stage).
CASES = {
    # Text-to-image, the default stage config (CFG 2.5, step cache 0.12).
    "t2i": ("hunyuan_image3_t2i_mars_seed42.png", ["--prompt", MARS_PROMPT]),
    # Image editing (VAE + ViT conditioning tokens) on a fixed input image.
    "edit": (
        "hunyuan_image3_edit_starry_seed42.png",
        ["--prompt", "Turn the sky into a starry night, keep everything else unchanged.",
         "--image", os.path.join(GOLDEN_DIR, EDIT_INPUT)],
    ),
    # Think on the backbone (AR, greedy), then the image conditioned on the text.
    "t2i_think": (
        "hunyuan_image3_t2i_think_mars_seed42.png",
        ["--prompt", MARS_PROMPT, "--bot-task", "think",
         "--stage-config", os.path.join(EXAMPLES, "hunyuan_image3_stage_ar.yaml")],
    ),
    # The tile-parallel VAE (vae_patch_parallel_size: 16).
    "t2i_vae_tiles": ("hunyuan_image3_t2i_vae_tiles_mars_seed42.png", ["--prompt", MARS_PROMPT]),
}


PACKAGES = ("torch", "libtorch-neuronx-lite", "neuronx-cc", "vllm", "vllm-omni", "vllm-neuron",
            "transformers", "diffusers")


def _sha256(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _manifest() -> dict:
    if not os.path.isfile(MANIFEST):
        return {"cases": {}, "inputs": {}}
    with open(MANIFEST) as handle:
        return json.load(handle)


def _conditions(case_args, model_path, trace_dir) -> dict:
    """What produced this run: request, stage config, checkpoint, environment, inputs."""
    import importlib.metadata

    import torch

    stage = case_args[case_args.index("--stage-config") + 1] if "--stage-config" in case_args else None
    with open(stage or os.path.join(EXAMPLES, "hunyuan_image3_stage.yaml")) as handle:
        stage_hash = hashlib.sha256(json.dumps(yaml.safe_load(handle), sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256()
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        with open(os.path.join(model_path, name), "rb") as handle:
            fingerprint.update(handle.read())
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    request = [a.replace(GOLDEN_DIR, "<goldens>").replace(EXAMPLES, "<examples>") for a in case_args]
    if stage and not stage.startswith(EXAMPLES):
        request[request.index(stage)] = "<generated>/" + os.path.basename(stage)
    inputs = torch.load(os.path.join(trace_dir, "inputs.pt"), weights_only=False)
    latents = torch.load(os.path.join(trace_dir, "step_00.pt"))["latents_in"]
    return {
        "request": ["--seed", "42", *request],
        "stage_config_sha256": stage_hash,
        "checkpoint_fingerprint": fingerprint.hexdigest(),
        "environment": versions,
        "input_ids_sha256": _tensor_sha256(inputs["input_ids"]),
        "initial_latents": latents,
    }


def _provenance() -> tuple[str, str]:
    """``(checkpoint revision, plugin commit)`` for a golden update, before anything runs."""
    revision = os.environ.get("HY3_MODEL_REVISION")
    assert revision, "HY3_UPDATE_GOLDEN=1 needs HY3_MODEL_REVISION, the checkpoint's revision"
    commit = os.environ.get("HY3_PLUGIN_COMMIT") or subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    assert commit, "set HY3_PLUGIN_COMMIT: the checkout is not a git tree"
    return revision, commit


def _update(manifest, case, golden_name, output, warm, conditions, provenance) -> None:
    """Write a reviewed run as the case's golden: the whole record first, then the files.

    The manifest entry is built from the run's output before anything in the golden
    directory changes, and the manifest is written last.
    """
    import torch

    revision, commit = provenance
    latents = conditions.pop("initial_latents")
    latents_sha = _tensor_sha256(latents)
    latents_file = f"initial_latents_{latents_sha[:12]}.pt"
    output_sha = _sha256(output)
    entry = {
        "golden": golden_name,
        "sha256": output_sha,
        "warm_seconds": round(max(warm), 2),
        "plugin_commit": commit,
        "model_revision": revision,
        "initial_latents": {"file": latents_file, "sha256": latents_sha},
        **conditions,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    }
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    torch.save(latents.contiguous().clone(), os.path.join(GOLDEN_DIR, latents_file))
    shutil.copyfile(output, os.path.join(GOLDEN_DIR, golden_name))
    if case == "t2i":
        shutil.copyfile(output, os.path.join(GOLDEN_DIR, EDIT_INPUT))
        manifest["inputs"] = {EDIT_INPUT: output_sha}
    manifest["cases"][case] = entry
    with open(MANIFEST + ".tmp", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(MANIFEST + ".tmp", MANIFEST)


def _check_conditions(case, entry, conditions) -> None:
    import torch

    latents = conditions.pop("initial_latents")
    archived = torch.load(os.path.join(GOLDEN_DIR, entry["initial_latents"]["file"]))
    assert _tensor_sha256(archived) == entry["initial_latents"]["sha256"], f"{case}: archived latents altered"
    assert _tensor_sha256(latents) == entry["initial_latents"]["sha256"], f"{case}: initial latents differ"
    for key in ("request", "stage_config_sha256", "checkpoint_fingerprint", "input_ids_sha256"):
        assert conditions[key] == entry[key], f"{case}: {key} {conditions[key]} != manifest {entry[key]}"
    changed = {k: (entry["environment"].get(k), v) for k, v in conditions["environment"].items()
               if entry["environment"].get(k) != v}
    if changed:
        print(f"{case}: package versions differ from the golden's (manifest, now): {changed}")


@pytest.mark.parametrize("case", list(CASES))
def test_generation_matches_golden(require_neuron_device, tmp_path, case):
    golden_name, case_args = CASES[case]
    golden = os.path.join(GOLDEN_DIR, golden_name)
    manifest = _manifest()
    entry = manifest["cases"].get(case)
    provenance = _provenance() if UPDATE else None
    if not UPDATE:
        assert entry is not None and os.path.isfile(golden), (
            f"{case}: no golden {golden_name} in {GOLDEN_DIR}; review an output and rerun with "
            "HY3_UPDATE_GOLDEN=1"
        )
        pinned = {golden_name: entry["sha256"], **manifest["inputs"]}
        for name in [golden_name] + ([EDIT_INPUT] if case == "edit" else []):
            assert _sha256(os.path.join(GOLDEN_DIR, name)) == pinned[name], f"{name} does not match manifest.json"
    if case == "t2i_vae_tiles":
        case_args = [*case_args, "--stage-config", _tile_vae_stage(tmp_path)]
    output, trace = tmp_path / f"{case}.png", tmp_path / "trace"
    log = run_example(
        "run.py",
        ["--model-path", require_neuron_device, "--seed", "42", "--output", str(output),
         "--profile", "--runs", "2", *case_args],
        str(tmp_path / f"{case}.log"),
        env={"HY3_TRACE_DIR": str(trace), "HY3_TRACE_STEPS": "0"},
    )
    warm = profile_times(log)
    assert len(warm) == 2, f"{case}: expected two timed requests in {tmp_path}/{case}.log"
    print(f"{case}: warm requests {warm} s")
    conditions = _conditions(case_args, require_neuron_device, trace)
    if UPDATE:
        _update(manifest, case, golden_name, output, warm, conditions, provenance)
        print(f"{case}: wrote golden {golden}")
        return
    _check_conditions(case, entry, conditions)
    actual, expected = load_rgb(str(output)), load_rgb(golden)
    score, db = ssim(actual, expected), psnr(actual, expected)
    print(f"{case}: SSIM {score:.4f}, PSNR {db:.2f} dB against {golden_name}")
    assert score >= SSIM_MIN, f"{case}: SSIM {score:.4f} < {SSIM_MIN} against {golden_name}"
    limit = entry["warm_seconds"] * (1 + CASE_LATENCY_TOLERANCE.get(case, LATENCY_TOLERANCE))
    assert min(warm) <= limit, f"{case}: warm request {min(warm):.2f} s > {limit:.2f} s (manifest {entry['warm_seconds']} s)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
