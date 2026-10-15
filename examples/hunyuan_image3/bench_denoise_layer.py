# SPDX-License-Identifier: Apache-2.0
"""Where does one HunyuanImage-3.0 denoise step spend its time, per decoder layer?

A warm 1024x1024 step is 0.90 s at TP32 over 32 layers -- ~28 ms per layer per rank --
and the routed MoE is assumed to dominate it. This times each piece of one layer on one
core, at one rank's real shapes (CFG batch 2 x 4097 tokens, 1 query head and 1 KV head,
2 of the 64 experts, a 96-wide slice of the shared expert), as its own compiled graph:

* ``attn``: QKV projection, 2D RoPE, QK RMSNorm, the two-bias denoise attention over
  [512 prompt | 4097 image] keys, and the NKI output projection;
* ``router``: the float32 router and its top-8 threshold;
* ``moe-dense``: both local experts evaluated on every token;
* ``moe-blockwise``: the default nkilib ``moe_cte`` path, mapping builder included;
* ``shared``: the shared expert's per-rank slice;
* ``norms``: the two RMSNorms and residual adds.

Collectives are not in any of these (one core, no group), so 32 x the sum, set against
the measured step, bounds what the two all-reduces per layer cost.

    python examples/hunyuan_image3/bench_denoise_layer.py
    python examples/hunyuan_image3/bench_denoise_layer.py --only moe-dense moe-blockwise
"""

import argparse
import os
import time

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "300")

import vllm_omni_neuron.bootstrap  # noqa: E402, F401  isort: skip  must precede vllm imports
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.nn.functional.gelu = torch.ops.aten.gelu.default

from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
)
from vllm_omni_neuron.lite_compat import initialize as initialize_lite  # noqa: E402

HIDDEN = 4096
HEAD_DIM = 128
NUM_EXPERTS = 64
TOP_K = 8
MOE_INTER = 3072
PIECES = (
    "attn",
    "router",
    "moe-dense",
    "moe-blockwise",
    "moe-mapping",
    "moe-kernel",
    "shared",
    "norms",
)

parser = argparse.ArgumentParser(description="Per-layer cost of a HunyuanImage-3.0 denoise step")
parser.add_argument("--tp-size", type=int, default=32)
parser.add_argument("--batch", type=int, default=2, help="CFG batch")
parser.add_argument("--image-len", type=int, default=4097, help="timestep token + image tokens")
parser.add_argument("--prefill-len", type=int, default=512)
parser.add_argument("--block-size", type=int, default=1024)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument(
    "--impl",
    default="shard_on_block",
    choices=["shard_on_block", "shard_on_i", "shard_on_i_hybrid"],
    help="moe_cte implementation for the moe-blockwise piece.",
)
parser.add_argument(
    "--num-static-block",
    type=int,
    default=None,
    help="shard_on_i_hybrid: blocks in the static loop; the rest run in the dynamic loop.",
)
parser.add_argument("--only", nargs="*", default=None, choices=PIECES)
args = parser.parse_args()


def _init_single_rank() -> None:
    """One-rank TP group: the blockwise mapping builder wants a GroupCoordinator."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29575")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    _init_single_rank._context = context
    initialize_model_parallel(tensor_model_parallel_size=1)


def _compile(fn, name):
    return torch.compile(
        fn,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": f"bench_denoise_{name}",
            "compiler_args": [
                "--model-type=transformer",
                "--auto-cast=none",
                "-O1",
                "--hbm-scratchpad-page-size=2048",
            ],
        },
    )


def _time(name, fn, *inputs):
    """Per-call time, synchronised per call through a scalar that depends on every output.

    Back-to-back calls without a sync overflow the runtime's execution queue
    (``status=7 Execution Queue Full``), and copying the full output back would time the
    transfer. Summing the output keeps all of it live; the sum itself is ~0.1 ms.
    """

    def wrapped(*xs):
        out = fn(*xs)
        return out.float().sum()

    compiled = _compile(wrapped, name.replace("-", "_"))
    start = time.perf_counter()
    compiled(*inputs).to("cpu")
    compile_seconds = time.perf_counter() - start
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        compiled(*inputs).to("cpu")
        times.append(time.perf_counter() - start)
    per_call = min(times)
    print(
        f"[layer] {name:<14} {per_call * 1e3:8.2f} ms/layer  x32 = {per_call * 32:6.3f} s/step"
        f"  (median {sorted(times)[len(times) // 2] * 1e3:.2f} ms, compile {compile_seconds:.0f}s)",
        flush=True,
    )
    return per_call


def main() -> None:
    from vllm_omni.platforms import current_omni_platform

    from vllm_omni_neuron.diffusion.models.hunyuan_image3 import hunyuan_image3_transformer as T

    initialize_lite()
    ensure_current_device_index()
    _init_single_rank()
    from vllm.distributed.parallel_state import get_tp_group

    device = current_omni_platform.get_torch_device(0)
    bf16 = torch.bfloat16
    local_experts = NUM_EXPERTS // args.tp_size
    shared_inter = MOE_INTER // args.tp_size
    b, s, p = args.batch, args.image_len, args.prefill_len
    tokens = b * s
    torch.manual_seed(0)

    def dev(*shape, dtype=bf16, scale=0.02):
        return (torch.randn(*shape, dtype=torch.float32) * scale).to(dtype).to(device)

    hidden = dev(b, s, HIDDEN, scale=1.0)
    print(
        f"one rank of TP{args.tp_size}: batch {b} x {s} tokens, prompt K/V {p}, "
        f"{local_experts} local experts, shared slice {shared_inter}, block {args.block_size}",
        flush=True,
    )
    results = {}
    want = set(args.only or PIECES)

    if "attn" in want:
        qkv_w = dev(HIDDEN, 3 * HEAD_DIM)
        o_w = dev(HEAD_DIM, HIDDEN)
        o_bias = torch.zeros(1, HIDDEN, dtype=bf16).to(device)
        qn, kn = dev(HEAD_DIM, scale=1.0), dev(HEAD_DIM, scale=1.0)
        cos, sin = (
            dev(b, s, HEAD_DIM // 2, dtype=torch.float32, scale=1.0),
            dev(b, s, HEAD_DIM // 2, dtype=torch.float32, scale=1.0),
        )
        pk, pv = dev(b, p, 1, HEAD_DIM, scale=1.0), dev(b, p, 1, HEAD_DIM, scale=1.0)
        zero = torch.zeros(1, 1, 1, p + s, dtype=torch.float32)
        tb, ib = zero.to(device), zero.to(device)

        def attn(h, qkv_w, o_w, o_bias, qn, kn, cos, sin, pk, pv, tb, ib):
            q, k, v = torch.tensor_split(torch.matmul(h, qkv_w), [HEAD_DIM, 2 * HEAD_DIM], -1)
            q = T.apply_rope_2d(q.reshape(b, s, 1, HEAD_DIM), cos, sin)
            k = T.apply_rope_2d(k.reshape(b, s, 1, HEAD_DIM), cos, sin)
            q, k = T.rms_norm(q, qn, 1e-5), T.rms_norm(k, kn, 1e-5)
            k = torch.cat((pk, k), 1)
            v = torch.cat((pv, v.reshape(b, s, 1, HEAD_DIM)), 1)
            a = T._denoise_attend(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), HEAD_DIM**-0.5, (tb, ib)
            )
            return T._hunyuan_o_proj(a, o_w, o_bias, True)

        results["attn"] = _time(
            "attn", attn, hidden, qkv_w, o_w, o_bias, qn, kn, cos, sin, pk, pv, tb, ib
        )

    flat = hidden.reshape(tokens, HIDDEN)
    gate_router = dev(HIDDEN, NUM_EXPERTS, dtype=torch.float32)

    def routing(t, g):
        probs = torch.softmax(torch.matmul(t.float(), g), dim=-1)
        thr = torch.topk(probs, TOP_K, dim=-1).values[..., -1:]
        dense = torch.where(probs >= thr, probs, torch.zeros_like(probs))
        return (dense / dense.sum(-1, keepdim=True).clamp(min=1e-8))[:, :local_experts].to(t.dtype)

    if "router" in want:
        results["router"] = _time("router", routing, flat, gate_router)

    gate_up = dev(local_experts, HIDDEN, 2, MOE_INTER)
    down = dev(local_experts, MOE_INTER, HIDDEN)
    # Realistic routing: the same top-8-of-64 distribution the model sees (12.5% live).
    weights = routing(flat.to("cpu"), gate_router.to("cpu")).to(device)

    reference = {}

    def check(label, fn):
        """Compare a routed-expert graph to float32 dense math on the host."""
        if "ref" not in reference:
            t_cpu, w_cpu = flat.to("cpu").float(), weights.to("cpu").float()
            gu, dn = gate_up.to("cpu").float(), down.to("cpu").float()
            reference["ref"] = sum(
                torch.matmul(F.silu(t_cpu @ gu[e, :, 0]) * (t_cpu @ gu[e, :, 1]), dn[e])
                * w_cpu[:, e : e + 1]
                for e in range(local_experts)
            )
        ref = reference["ref"]
        got = _compile(fn, f"moe_check_{label.replace('-', '_')}")(flat, weights, gate_up, down)
        got = got.to("cpu").float()
        worst = ((got - ref).abs().max() / ref.abs().max()).item()
        rel_l2 = (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()
        print(
            f"[check] {label}: vs float32 dense, max {worst:.3e}, relative L2 {rel_l2:.3e}",
            flush=True,
        )

    if "moe-dense" in want:

        def moe_dense(t, w, gate_up, down):
            out = None
            for e in range(local_experts):
                g = F.silu(torch.matmul(t, gate_up[e, :, 0, :])) * torch.matmul(
                    t, gate_up[e, :, 1, :]
                )
                y = torch.matmul(g, down[e]) * w[:, e : e + 1]
                out = y if out is None else out + y
            return out

        results["moe-dense"] = _time("moe-dense", moe_dense, flat, weights, gate_up, down)
        check("moe-dense", moe_dense)

    if "moe-blockwise" in want:
        import vllm_neuron.functional as NF
        from nkilib.core.moe.moe_cte.moe_cte import ExpertAffinityScaleMode, MoECTEImplementation
        from nkilib.core.utils.common_types import ActFnType

        bs = args.block_size
        padded = -(-tokens // bs) * bs

        def moe_blockwise(t, w, gate_up, down):
            t = F.pad(t, (0, 0, 0, padded - tokens))
            w = F.pad(w, (0, 0, 0, padded - tokens))
            mask = torch.arange(padded, device=t.device) < tokens
            aff, pos, b2e, cond = NF.build_blockwise_mapping(
                expert_affinities=w,
                num_local_experts=local_experts,
                num_experts_per_token=TOP_K,
                block_size=bs,
                moe_group=get_tp_group(),
                tp_degree=1,
                padding_mask=mask,
            )
            extra = {}
            if args.impl == "shard_on_i_hybrid":
                extra["num_static_block"] = args.num_static_block
            out = NF.moe_cte(
                implementation=getattr(MoECTEImplementation, args.impl),
                conditions=cond,
                hidden_states=t,
                expert_affinities_masked=aff,
                gate_up_proj_weight=gate_up,
                down_proj_weight=down,
                activation_function=ActFnType.SiLU,
                block_size=bs,
                token_position_to_id=pos.to(dtype=torch.int32),
                block_to_expert=b2e.to(dtype=torch.int32),
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                skip_token=True,
                is_tensor_update_accumulating=True,
                **extra,
            )
            out = out[0] if isinstance(out, tuple) else out
            return out[:tokens]

        label = "moe-blockwise" if args.impl == "shard_on_block" else f"moe-{args.impl}"
        results["moe-blockwise"] = _time(label, moe_blockwise, flat, weights, gate_up, down)
        check(label, moe_blockwise)

    if "moe-mapping" in want or "moe-kernel" in want:
        import vllm_neuron.functional as NF
        from nkilib.core.moe.moe_cte.moe_cte import ExpertAffinityScaleMode, MoECTEImplementation
        from nkilib.core.utils.common_types import ActFnType

        bs = args.block_size
        padded = -(-tokens // bs) * bs
        padded_tokens = F.pad(flat, (0, 0, 0, padded - tokens))
        padded_weights = F.pad(weights, (0, 0, 0, padded - tokens))

        def mapping(w):
            mask = torch.arange(padded, device=w.device) < tokens
            aff, pos, b2e, cond = NF.build_blockwise_mapping(
                expert_affinities=w,
                num_local_experts=local_experts,
                num_experts_per_token=TOP_K,
                block_size=bs,
                moe_group=get_tp_group(),
                tp_degree=1,
                padding_mask=mask,
            )
            return aff, pos.to(dtype=torch.int32), b2e.to(dtype=torch.int32), cond

        if "moe-mapping" in want:
            results["moe-mapping"] = _time("moe-mapping", lambda w: mapping(w)[0], padded_weights)

        if "moe-kernel" in want:
            # The mapping computed once (eagerly through its own graph) and fed in as data.
            mapped = _compile(mapping, "moe_mapping_once")(padded_weights)

            def kernel_only(t, aff, pos, b2e, cond, gate_up, down):
                out = NF.moe_cte(
                    implementation=MoECTEImplementation.shard_on_block,
                    conditions=cond,
                    hidden_states=t,
                    expert_affinities_masked=aff,
                    gate_up_proj_weight=gate_up,
                    down_proj_weight=down,
                    activation_function=ActFnType.SiLU,
                    block_size=bs,
                    token_position_to_id=pos,
                    block_to_expert=b2e,
                    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                    skip_token=True,
                    is_tensor_update_accumulating=True,
                )
                return out[0] if isinstance(out, tuple) else out

            results["moe-kernel"] = _time(
                "moe-kernel", kernel_only, padded_tokens, *mapped, gate_up, down
            )
            print(
                f"[info] blocks {tuple(mapped[2].shape)}, conditions {tuple(mapped[3].shape)}",
                flush=True,
            )

    if "shared" in want:
        sg, su, sd = dev(HIDDEN, shared_inter), dev(HIDDEN, shared_inter), dev(shared_inter, HIDDEN)

        def shared(t, sg, su, sd):
            t = t.unsqueeze(0)
            return torch.matmul(F.silu(torch.matmul(t, sg)) * torch.matmul(t, su), sd).squeeze(0)

        results["shared"] = _time("shared", shared, flat, sg, su, sd)

    if "norms" in want:
        n1, n2 = dev(HIDDEN, scale=1.0), dev(HIDDEN, scale=1.0)
        other = dev(b, s, HIDDEN, scale=1.0)

        def norms(h, o, n1, n2):
            x = h + T.rms_norm(h, n1, 1e-5)
            return x + T.rms_norm(x + o, n2, 1e-5)

        results["norms"] = _time("norms", norms, hidden, other, n1, n2)

    base = sum(
        v
        for k, v in results.items()
        if k not in ("moe-dense", "moe-blockwise", "moe-mapping", "moe-kernel")
    )
    for moe in ("moe-dense", "moe-blockwise"):
        if moe in results:
            print(
                f"[summary] compute per step with {moe}: {(base + results[moe]) * 32:.3f} s "
                f"(MoE {results[moe] * 32:.3f} s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
