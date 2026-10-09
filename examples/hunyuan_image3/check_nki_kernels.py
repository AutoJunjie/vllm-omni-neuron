# SPDX-License-Identifier: Apache-2.0
"""On-device check of the NKI kernels the HunyuanImage-3.0 backbone uses.

Compiles each kernel on its own, at the shapes the real model runs, and compares it
against the torch math the fallback path computes. Minutes instead of the hours a full
cold DiT compile takes, which makes this the right first thing to run on a new node:
a kernel whose operand convention or tiling assumption is off shows up here rather than
as washed-out pixels after a long build.

    python examples/hunyuan_image3/check_nki_kernels.py
    python examples/hunyuan_image3/check_nki_kernels.py --tp-size 32

For the SwiGLU MLP it reports the error for *both* possible gate/up assignments, so the
output states which operand nkilib puts through the activation rather than assuming it.
"""

import argparse
import os

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_CC_FLAGS", "-O1 --hbm-scratchpad-page-size=2048")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
# Bound every device execution: on a managed cluster an unresponsive core gets the whole
# node replaced within seconds, which destroys the state you would debug from.
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "120")

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# torch_neuronx / Lite patch F.gelu with a non-traceable C wrapper.
torch.nn.functional.gelu = torch.ops.aten.gelu.default

from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

from vllm_omni_neuron.diffusion.models.hunyuan_image3 import (  # noqa: E402
    hunyuan_image3_transformer as hyt,
)
from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
    initialize as initialize_lite,
)

parser = argparse.ArgumentParser(description="NKI kernel checks for HunyuanImage-3.0")
parser.add_argument("--tp-size", type=int, default=32, help="TP degree the shapes follow")
parser.add_argument("--tokens", type=int, default=8194, help="CFG batch x (1 + image tokens)")
parser.add_argument("--prefill-len", type=int, default=512)
parser.add_argument("--tolerance", type=float, default=2e-2)
parser.add_argument(
    "--only",
    default="o_proj,attention,mlp",
    help="Comma-separated subset of o_proj,attention,mlp. A run that first launched the\n"
    "MLP kernel stopped producing output, so run it on its own once the others are\n"
    "known good — then the process under suspicion is the only one on the device.",
)
args = parser.parse_args()

HIDDEN = 4096
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
MOE_INTERMEDIATE = 3072


def _device() -> torch.device:
    from vllm_omni.platforms import current_omni_platform

    return current_omni_platform.get_torch_device(0)


def _compile(fn, name: str):
    return torch.compile(
        fn,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": name,
            "compiler_args": [
                "--model-type=transformer",
                "--auto-cast=none",
                "-O1",
                "--hbm-scratchpad-page-size=2048",
            ],
        },
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    scale = expected.abs().max().item()
    return (actual - expected).abs().max().item() / max(scale, 1e-12)


def check_swiglu_mlp(device) -> dict[str, float]:
    """nkilib SwiGLU MLP vs torch, under both gate/up operand assignments."""
    torch.manual_seed(0)
    tokens = args.tokens
    hidden = torch.randn(1, tokens, HIDDEN, dtype=torch.bfloat16) * 0.05
    gate_w = torch.randn(HIDDEN, MOE_INTERMEDIATE, dtype=torch.bfloat16) * 0.02
    up_w = torch.randn(HIDDEN, MOE_INTERMEDIATE, dtype=torch.bfloat16) * 0.02
    down_w = torch.randn(MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16) * 0.02
    inner_bias = torch.zeros(1, MOE_INTERMEDIATE, dtype=torch.bfloat16)
    outer_bias = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)

    def kernel(h, g, u, d, bi, bo):
        return hyt._hunyuan_nki_swiglu_mlp(h, g, u, d, bi, bi, bo)

    compiled = _compile(kernel, "hunyuan_check_swiglu_mlp")
    actual = compiled(
        hidden.to(device),
        gate_w.to(device),
        up_w.to(device),
        down_w.to(device),
        inner_bias.to(device),
        outer_bias.to(device),
    ).to("cpu")

    hidden_f = hidden.float()
    activate_gate = torch.matmul(
        F.silu(torch.matmul(hidden_f, gate_w.float())) * torch.matmul(hidden_f, up_w.float()),
        down_w.float(),
    )
    activate_up = torch.matmul(
        F.silu(torch.matmul(hidden_f, up_w.float())) * torch.matmul(hidden_f, gate_w.float()),
        down_w.float(),
    )
    return {
        "activation_on_gate_operand": _relative_error(actual, activate_gate),
        "activation_on_up_operand": _relative_error(actual, activate_up),
    }


def check_causal_attention(device) -> dict[str, float]:
    """attention_cte (causal, d-major out) vs a torch masked softmax, at prefill shapes."""
    torch.manual_seed(0)
    heads = max(1, HEADS // args.tp_size)
    batch = 2  # CFG batch
    seq = args.prefill_len
    q = torch.randn(batch, heads, seq, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(batch, heads, seq, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(batch, heads, seq, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    scale = HEAD_DIM**-0.5

    def kernel(qq, kk, vv):
        scaled = qq * scale
        b, n, s, d = scaled.shape
        out = hyt._hunyuan_nki_causal_attention(
            scaled.reshape(b * n, s, d).contiguous(),
            kk.reshape(b * n, s, d).contiguous(),
            vv.reshape(b * n, s, d).contiguous(),
        )
        return out.reshape(b, n, d, s)

    compiled = _compile(kernel, "hunyuan_check_causal_attention")
    actual = compiled(q.to(device), k.to(device), v.to(device)).to("cpu")

    causal = torch.ones(seq, seq, dtype=torch.bool).tril()
    scores = torch.matmul(q.float() * scale, k.float().transpose(-2, -1))
    scores = scores.masked_fill(~causal, float("-inf"))
    expected = torch.matmul(torch.softmax(scores, dim=-1), v.float()).transpose(-2, -1)
    return {"causal_attention": _relative_error(actual, expected)}


def check_output_projection(device) -> dict[str, float]:
    """output_projection_cte vs matmul, at this rank's attention-output shape."""
    torch.manual_seed(0)
    heads = max(1, HEADS // args.tp_size)
    batch = 2
    seq = args.tokens // batch
    active = torch.randn(batch, heads, HEAD_DIM, seq, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(heads * HEAD_DIM, HIDDEN, dtype=torch.bfloat16) * 0.02
    bias = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)

    def kernel(a, w, b):
        return hyt._hunyuan_nki_o_proj(a, w, b)

    compiled = _compile(kernel, "hunyuan_check_output_projection")
    actual = compiled(active.to(device), weight.to(device), bias.to(device)).to("cpu")

    flat = active.float().reshape(batch, heads * HEAD_DIM, seq).transpose(1, 2)
    expected = torch.matmul(flat, weight.float())
    return {"output_projection": _relative_error(actual, expected)}


def main() -> None:
    initialize_lite()
    ensure_current_device_index()
    device = _device()
    print(f"device={device} tp_size={args.tp_size} tokens={args.tokens}")

    selected = [name.strip() for name in args.only.split(",") if name.strip()]
    checks = {
        "o_proj": check_output_projection,
        "attention": check_causal_attention,
        "mlp": check_swiglu_mlp,
    }
    unknown = set(selected) - set(checks)
    if unknown:
        raise SystemExit(f"unknown --only entries: {sorted(unknown)}")

    results: dict[str, float] = {}
    # Report each check as it lands: a later kernel failing (or hanging) should not hide
    # the numbers the earlier ones already produced.
    for name in selected:
        partial = checks[name](device)
        results.update(partial)
        for key, error in partial.items():
            print(f"  {key:34s} {error:.4e}", flush=True)

    mlp_errors = [
        results[key]
        for key in ("activation_on_gate_operand", "activation_on_up_operand")
        if key in results
    ]
    if mlp_errors:
        gate_error, up_error = mlp_errors
        convention = "gate operand" if gate_error < up_error else "up operand"
        print(
            f"\nnkilib applies the activation to the {convention}; the plugin passes the "
            "checkpoint's second gate_and_up_proj chunk as `gate`."
        )

    failures = {
        name: error
        for name, error in results.items()
        if name not in ("activation_on_gate_operand", "activation_on_up_operand")
        and error > args.tolerance
    }
    if mlp_errors and min(mlp_errors) > args.tolerance:
        failures["swiglu_mlp"] = min(mlp_errors)
    if failures:
        raise SystemExit(f"FAIL: {failures} (tolerance {args.tolerance})")
    print("OK")


if __name__ == "__main__":
    main()
