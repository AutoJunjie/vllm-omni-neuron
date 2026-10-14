# SPDX-License-Identifier: Apache-2.0
"""Compare denoise-step traces recorded by the pipeline's ``_StepTrace``.

Every trace directory holds ``step_NN.pt`` with the step's input latents and its
predictions. Traces taken with teacher forcing share their input latents, so each
prediction is compared with another on identical inputs:

    python examples/hunyuan_image3/compare_step_traces.py \\
        --reference cpu_fp32=/workspace/out/trace/cpu_fp32 \\
        hybrid=/workspace/out/trace/hybrid dense=/workspace/out/trace/dense \\
        block=/workspace/out/trace/block

For every step present in both, reports the relative L2 error of the combined (CFG)
prediction -- what the scheduler consumes -- and, with CFG, of the conditional and
unconditional branches, plus the cosine similarity of the combined one and the error of
the scheduler's updated latents.

``--max-rel-l2`` makes it a check, exit 1 unless every compared trace holds every
required step -- the reference's, or ``--steps`` -- with the same timestep, input latents
and tensor shapes as the reference, nothing non-finite, and its combined prediction and
updated latents within the bound.
"""

import argparse
import glob
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True, help="name=dir of the reference trace")
parser.add_argument("traces", nargs="+", help="name=dir of each trace to compare")
parser.add_argument("--max-rel-l2", type=float, default=None, help="fail above this error")
parser.add_argument("--steps", default=None, help="with --max-rel-l2: compare only these, e.g. 0,25,49")
args = parser.parse_args()


def load(spec):
    name, path = spec.split("=", 1)
    steps = {}
    for f in sorted(glob.glob(os.path.join(path, "step_*.pt"))):
        steps[int(os.path.basename(f)[5:7])] = torch.load(f)
    return name, steps


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def mean(values):
    return sum(values) / len(values)


FIELDS = ("timestep", "latents_in", "pred_raw", "pred")


def check(trace, ref, required):
    """What makes ``trace`` not comparable with ``ref`` at the ``required`` steps."""
    problems = [f"step {k} missing" for k in required if k not in trace or k not in ref]
    if args.steps is None and set(trace) != set(ref):
        problems.append(f"steps {sorted(trace)} differ from the reference's {sorted(ref)}")
    for k in (k for k in required if k in trace and k in ref):
        got, want = trace[k], ref[k]
        fields = FIELDS + (("latents_out",) if "latents_out" in got or "latents_out" in want else ())
        for field in fields:
            if field not in got or field not in want:
                problems.append(f"step {k}: no {field}")
            elif got[field].shape != want[field].shape:
                problems.append(f"step {k}: {field} shape {tuple(got[field].shape)} vs {tuple(want[field].shape)}")
            elif not (torch.isfinite(got[field]).all() and torch.isfinite(want[field]).all()):
                problems.append(f"step {k}: {field} not finite")
        for field in ("timestep", "latents_in"):
            if field in got and field in want and not torch.equal(got[field], want[field]):
                problems.append(f"step {k}: {field} differs from the reference's")
    return problems


ref_name, ref = load(args.reference)
print(f"reference {ref_name}: steps {sorted(ref)}")
strict = args.max_rel_l2 is not None
required = sorted(int(s) for s in args.steps.split(",")) if args.steps else sorted(ref)
failed = False
for spec in args.traces:
    name, trace = load(spec)
    if strict:
        problems = check(trace, ref, required)
        if problems:
            print(f"{name}: FAIL, not comparable with {ref_name}: " + "; ".join(problems[:8]))
            failed = True
            continue
    common = required if strict else sorted(set(trace) & set(ref))
    if not common:
        print(f"{name}: no steps in common with {ref_name}")
        failed = True
        continue
    same_inputs = all(torch.equal(trace[k]["latents_in"], ref[k]["latents_in"]) for k in common)
    errs, branches, cos, outs = [], [], [], []
    for k in common:
        got, want = trace[k], ref[k]
        errs.append(rel_l2(got["pred"], want["pred"]))
        if got["pred_raw"].shape[0] == 2 and want["pred_raw"].shape[0] == 2:
            branches.append([rel_l2(a, b) for a, b in zip(got["pred_raw"].chunk(2), want["pred_raw"].chunk(2))])
        cos.append(
            torch.nn.functional.cosine_similarity(
                got["pred"].flatten(), want["pred"].flatten(), dim=0
            ).item()
        )
        if "latents_out" in got and "latents_out" in want:
            outs.append(rel_l2(got["latents_out"], want["latents_out"]))
    line = (
        f"{name} vs {ref_name}: {len(common)} steps, identical inputs {same_inputs}; "
        f"combined rel-L2 mean {mean(errs):.3e} max {max(errs):.3e}; cosine min {min(cos):.6f}"
    )
    if branches:
        line += f"; cond mean {mean([b[0] for b in branches]):.3e}; uncond mean {mean([b[1] for b in branches]):.3e}"
    if outs:
        line += f"; updated latents rel-L2 max {max(outs):.3e}"
    print(line)
    if len(common) <= 6:
        for k, e in zip(common, errs):
            print(f"    step {k:02d}: combined {e:.3e}")
    worst = max(errs + outs)
    if strict and not worst <= args.max_rel_l2:  # also catches a NaN error
        print(f"    FAIL: rel-L2 {worst:.3e} above --max-rel-l2 {args.max_rel_l2}")
        failed = True
raise SystemExit(1 if failed and strict else 0)
