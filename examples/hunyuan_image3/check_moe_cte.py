# SPDX-License-Identifier: Apache-2.0
"""On-device check of the blockwise CTE MoE kernel at HunyuanImage-3.0's shapes.

The dense SwiGLU MLP kernel caps at 256 tokens per launch
(`[NCC_INKI016] Stack out of memory` above that), which would mean 32 launches per
expert for this model's 8194-token step. `NF.moe_cte` is the right tool instead: it
blocks internally at `block_size`, takes the router's dense `[T, E_local]` affinities
directly, and only computes the (token, expert) pairs the router actually selected — so
it replaces the dense-over-all-local-experts loop *and* drops roughly 8x of its FLOPs.

This script checks that claim in isolation, before the model is restructured around it:
build the blockwise mapping from a synthetic top-k routing, run the kernel, and compare
against the torch MoE math. Run it on a node with auto-recovery disabled.

    python examples/hunyuan_image3/check_moe_cte.py
    python examples/hunyuan_image3/check_moe_cte.py --tp-size 16 --tokens 8194
"""

import argparse
import os

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_CC_FLAGS", "-O1 --hbm-scratchpad-page-size=2048")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
# Bound every device execution: an unresponsive core gets a managed node replaced.
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "120")

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.nn.functional.gelu = torch.ops.aten.gelu.default

import vllm_neuron.functional as NF  # noqa: E402
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation  # noqa: E402
from nkilib.core.utils.common_types import ActFnType  # noqa: E402
from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
    initialize as initialize_lite,
)

HIDDEN = 4096
NUM_EXPERTS = 64
TOP_K = 8
MOE_INTERMEDIATE = 3072

parser = argparse.ArgumentParser(description="Blockwise CTE MoE kernel check")
parser.add_argument("--tp-size", type=int, default=32, help="EP degree (experts // tp)")
parser.add_argument("--tokens", type=int, default=8194, help="CFG batch x (1 + image tokens)")
parser.add_argument("--block-size", type=int, default=256)
parser.add_argument("--tolerance", type=float, default=3e-2)
args = parser.parse_args()


def _init_single_rank() -> None:
    """One-rank TP group: moe_cte at tp_degree=1 does no collectives, but the mapping
    builder still wants a GroupCoordinator."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29573")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    _init_single_rank._context = context  # keep alive
    initialize_model_parallel(tensor_model_parallel_size=1)


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


def _torch_moe(hidden, affinities, gate_w, up_w, down_w):
    """Reference: sum over local experts of affinity * down(silu(x @ gate) * (x @ up))."""
    out = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(gate_w.shape[0]):
        gated = F.silu(torch.matmul(hidden, gate_w[expert])) * torch.matmul(
            hidden, up_w[expert]
        )
        out = out + torch.matmul(gated, down_w[expert]) * affinities[:, expert : expert + 1]
    return out


def main() -> None:
    initialize_lite()
    ensure_current_device_index()
    _init_single_rank()

    from vllm.distributed.parallel_state import get_tp_group

    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)
    num_local_experts = NUM_EXPERTS // args.tp_size
    tokens = args.tokens
    print(
        f"device={device} ep_degree={args.tp_size} local_experts={num_local_experts} "
        f"tokens={tokens} block_size={args.block_size}"
    )

    torch.manual_seed(0)
    # Router: each token picks TOP_K of NUM_EXPERTS globally, so on this rank only the
    # selected subset of its local experts is non-zero.
    logits = torch.randn(tokens, NUM_EXPERTS, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    threshold = torch.topk(probs, TOP_K, dim=-1).values[..., -1:]
    dense = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    dense = dense / dense.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    affinities = dense[:, :num_local_experts].contiguous()
    selected = int((affinities != 0).sum().item())
    print(f"local (token, expert) pairs selected: {selected} of {tokens * num_local_experts}")

    hidden = (torch.randn(tokens, HIDDEN, dtype=torch.bfloat16) * 0.05).contiguous()
    gate_w = torch.randn(num_local_experts, HIDDEN, MOE_INTERMEDIATE, dtype=torch.bfloat16) * 0.02
    up_w = torch.randn(num_local_experts, HIDDEN, MOE_INTERMEDIATE, dtype=torch.bfloat16) * 0.02
    down_w = torch.randn(num_local_experts, MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16) * 0.02
    # The kernel wants gate and up fused as [E, H, 2, I].
    gate_up = torch.stack((gate_w, up_w), dim=2).contiguous()

    def moe(hidden_d, affinities_d, gate_up_d, down_d):
        (
            affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=affinities_d,
            num_local_experts=num_local_experts,
            num_experts_per_token=TOP_K,
            block_size=args.block_size,
            moe_group=get_tp_group(),
            tp_degree=1,
        )
        return NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_d,
            expert_affinities_masked=affinities_masked,
            gate_up_proj_weight=gate_up_d,
            down_proj_weight=down_d,
            activation_function=ActFnType.SiLU,
            block_size=args.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            skip_token=True,
            is_tensor_update_accumulating=True,
        )

    compiled = _compile(moe, "hunyuan_check_moe_cte")
    actual = compiled(
        hidden.to(device),
        affinities.to(device),
        gate_up.to(device),
        down_w.to(device),
    )
    actual = (actual[0] if isinstance(actual, tuple) else actual).to("cpu").float()

    expected = _torch_moe(
        hidden.float(), affinities, gate_w.float(), up_w.float(), down_w.float()
    )
    if actual.shape != expected.shape:
        print(f"shape mismatch: kernel {tuple(actual.shape)} vs torch {tuple(expected.shape)}")
        actual = actual.reshape(expected.shape)

    scale = expected.abs().max().item()
    error = (actual - expected).abs().max().item() / max(scale, 1e-12)
    print(f"\nmoe_cte relative max error vs torch: {error:.4e}  (signal {scale:.4e})")
    if error > args.tolerance:
        raise SystemExit(f"FAIL: {error:.4e} > tolerance {args.tolerance}")
    print("OK")


if __name__ == "__main__":
    main()
