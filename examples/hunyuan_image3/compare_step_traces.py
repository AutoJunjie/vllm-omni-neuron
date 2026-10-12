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
prediction -- what the scheduler consumes -- and of the conditional and unconditional
branches, plus the cosine similarity of the combined one.
"""

import argparse
import glob
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True, help="name=dir of the reference trace")
parser.add_argument("traces", nargs="+", help="name=dir of each trace to compare")
args = parser.parse_args()


def load(spec):
    name, path = spec.split("=", 1)
    steps = {}
    for f in sorted(glob.glob(os.path.join(path, "step_*.pt"))):
        steps[int(os.path.basename(f)[5:7])] = torch.load(f)
    return name, steps


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


ref_name, ref = load(args.reference)
print(f"reference {ref_name}: steps {sorted(ref)}")
for spec in args.traces:
    name, trace = load(spec)
    common = sorted(set(trace) & set(ref))
    if not common:
        print(f"{name}: no steps in common with {ref_name}")
        continue
    same_inputs = all(torch.equal(trace[k]["latents_in"], ref[k]["latents_in"]) for k in common)
    errs, cond, uncond, cos = [], [], [], []
    for k in common:
        got, want = trace[k], ref[k]
        errs.append(rel_l2(got["pred"], want["pred"]))
        c, u = got["pred_raw"].chunk(2)
        rc, ru = want["pred_raw"].chunk(2)
        cond.append(rel_l2(c, rc))
        uncond.append(rel_l2(u, ru))
        cos.append(
            torch.nn.functional.cosine_similarity(
                got["pred"].flatten(), want["pred"].flatten(), dim=0
            ).item()
        )
    print(
        f"{name} vs {ref_name}: {len(common)} steps, identical inputs {same_inputs}; "
        f"combined rel-L2 mean {sum(errs) / len(errs):.3e} max {max(errs):.3e}; "
        f"cond mean {sum(cond) / len(cond):.3e}; uncond mean {sum(uncond) / len(uncond):.3e}; "
        f"cosine min {min(cos):.6f}"
    )
    if len(common) <= 6:
        for k, e, c, u in zip(common, errs, cond, uncond):
            print(f"    step {k:02d}: combined {e:.3e}  cond {c:.3e}  uncond {u:.3e}")
