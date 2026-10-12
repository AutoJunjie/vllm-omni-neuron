# SPDX-License-Identifier: Apache-2.0
"""Rank-by-rank check of the HunyuanImage-3.0 weight loaders.

The numerics test runs at TP1, where every sharding loader degenerates to the full
tensor. This one drives the loader transforms directly for each rank of a real TP degree
and asserts the shards tile the checkpoint tensor exactly, in the order the forward pass
assumes. No distributed setup and no device needed.

What it pins down:

* fused ``qkv_proj`` — the checkpoint groups rows per KV head (``g`` query heads, then
  that group's K and V head), so rank ``r`` must get query heads
  ``r * heads_per_rank ...`` *and* the KV head those query heads belong to. With
  ``num_key_value_heads < tp_size`` the KV heads are replicated, and the replication
  grouping has to line up with the query-head split or attention silently mixes groups.
* row-parallel ``o_proj`` — each rank's input columns must match its query heads.
* fused ``gate_and_up_proj`` — chunk 0 is the linear branch, chunk 1 the gated one, and
  the shared MLP shards the intermediate dim on top of that chunking.

    python test/unit/test_hunyuan_image3_sharding.py
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import torch  # noqa: E402

from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (  # noqa: E402
    _fused_chunk_sharded_loader,
    _interleaved_qkv_loader,
    _stacked_down_loader,
    _stacked_gate_up_loader,
    _transpose_loader,
)
from vllm_neuron.utils.weight_loader import sharding_weight_loader  # noqa: E402

HIDDEN = 4096
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
MOE_INTERMEDIATE = 3072


class FakeSlice:
    """Minimal stand-in for safetensors' ``PySafeSlice`` over a torch tensor."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getitem__(self, item):
        return self._tensor[item]

    def get_shape(self):
        return list(self._tensor.shape)


def _qkv_checkpoint() -> torch.Tensor:
    """A ``[kv_heads * (g + 2) * head_dim, hidden]`` tensor whose rows are self-identifying.

    Row ``(group, slot, d)`` is filled with ``group * 100 + slot``, so a shard can be
    checked against the (group, slot) it should have come from.
    """
    groups = HEADS // KV_HEADS
    weight = torch.zeros(KV_HEADS, groups + 2, HEAD_DIM, HIDDEN)
    for group in range(KV_HEADS):
        for slot in range(groups + 2):
            weight[group, slot] = group * 100 + slot
    return weight.reshape(-1, HIDDEN)


def test_interleaved_qkv_loader() -> None:
    weight = _qkv_checkpoint()
    groups = HEADS // KV_HEADS

    for tp_size in (16, 32):
        heads_per_rank = HEADS // tp_size
        kv_per_rank = max(1, KV_HEADS // tp_size)
        loader = _interleaved_qkv_loader(
            num_kv_heads=KV_HEADS,
            num_kv_groups=groups,
            head_dim=HEAD_DIM,
            tp_size=tp_size,
            num_heads_per_rank=heads_per_rank,
            num_kv_heads_per_rank=kv_per_rank,
        )
        for rank in range(tp_size):
            shard = loader.load([FakeSlice(weight)], rank)
            assert shard.shape == (
                HIDDEN,
                (heads_per_rank + 2 * kv_per_rank) * HEAD_DIM,
            ), shard.shape
            # Parameter layout is [hidden, q | k | v]; transpose back to inspect rows.
            rows = shard.T
            q_rows = rows[: heads_per_rank * HEAD_DIM].reshape(heads_per_rank, HEAD_DIM, HIDDEN)
            k_rows = rows[
                heads_per_rank * HEAD_DIM : (heads_per_rank + kv_per_rank) * HEAD_DIM
            ].reshape(kv_per_rank, HEAD_DIM, HIDDEN)
            v_rows = rows[(heads_per_rank + kv_per_rank) * HEAD_DIM :].reshape(
                kv_per_rank, HEAD_DIM, HIDDEN
            )

            for local, head in enumerate(
                range(rank * heads_per_rank, (rank + 1) * heads_per_rank)
            ):
                expected = (head // groups) * 100 + (head % groups)
                actual = q_rows[local].unique()
                assert actual.numel() == 1 and int(actual.item()) == expected, (
                    f"tp{tp_size} rank{rank}: query head {head} got {actual.tolist()}, "
                    f"expected {expected}"
                )

            # Every query head on this rank must belong to the KV head(s) it loaded.
            query_groups = {
                head // groups
                for head in range(rank * heads_per_rank, (rank + 1) * heads_per_rank)
            }
            kv_start = (
                rank * kv_per_rank if KV_HEADS >= tp_size else (rank * KV_HEADS) // tp_size
            )
            loaded_groups = set(range(kv_start, kv_start + kv_per_rank))
            assert query_groups <= loaded_groups, (
                f"tp{tp_size} rank{rank}: query heads span KV groups {sorted(query_groups)} "
                f"but loaded {sorted(loaded_groups)}"
            )
            for local, group in enumerate(loaded_groups):
                k_value = k_rows[local].unique()
                v_value = v_rows[local].unique()
                assert int(k_value.item()) == group * 100 + groups, k_value
                assert int(v_value.item()) == group * 100 + groups + 1, v_value
    print("interleaved qkv loader OK")


def test_o_proj_loader_matches_query_heads() -> None:
    """Row-parallel o_proj columns must line up with each rank's query heads."""
    weight = torch.arange(HIDDEN * HIDDEN, dtype=torch.float32).reshape(HIDDEN, HIDDEN)
    for tp_size in (16, 32):
        q_size = (HEADS // tp_size) * HEAD_DIM
        loader = sharding_weight_loader(
            shard_dim=0, shard_size=q_size, num_shards=tp_size, is_storage_transposed=True
        )
        rebuilt = torch.cat(
            [loader.load([FakeSlice(weight)], rank) for rank in range(tp_size)], dim=0
        )
        assert torch.equal(rebuilt, weight.T), f"tp{tp_size} o_proj shards do not tile"
    print("o_proj loader OK")


def test_stacked_expert_loaders() -> None:
    """Expert weights must land in the kernel's [E, H, 2, I] / [E, I, H] layout.

    Two orderings have to be right at once and neither fails loudly: the checkpoint's
    fused tensor is [up | gate] (the reference computes ``down(chunk0 * silu(chunk1))``)
    while the kernel's length-2 axis is (gate, up) — index 0 is the branch the activation
    applies to. Getting them crossed swaps SiLU onto the wrong projection.
    """
    experts = 2
    gate_up_slices = []
    down_slices = []
    for expert in range(experts):
        # Mark the branches distinguishably per expert: up = 10*e + 1, gate = 10*e + 2.
        gate_up_slices.append(
            FakeSlice(
                torch.cat(
                    (
                        torch.full((MOE_INTERMEDIATE, HIDDEN), 10.0 * expert + 1.0),
                        torch.full((MOE_INTERMEDIATE, HIDDEN), 10.0 * expert + 2.0),
                    ),
                    dim=0,
                )
            )
        )
        down_slices.append(
            FakeSlice(torch.full((HIDDEN, MOE_INTERMEDIATE), 100.0 + expert))
        )

    stacked = _stacked_gate_up_loader(MOE_INTERMEDIATE).load(gate_up_slices, 0)
    assert stacked.shape == (experts, HIDDEN, 2, MOE_INTERMEDIATE), stacked.shape
    for expert in range(experts):
        gate = stacked[expert, :, 0, :].unique()
        up = stacked[expert, :, 1, :].unique()
        assert gate.numel() == 1 and float(gate.item()) == 10.0 * expert + 2.0, (
            f"expert {expert}: kernel index 0 must hold the gated branch (chunk 1)"
        )
        assert up.numel() == 1 and float(up.item()) == 10.0 * expert + 1.0, (
            f"expert {expert}: kernel index 1 must hold the linear branch (chunk 0)"
        )

    down = _stacked_down_loader().load(down_slices, 0)
    assert down.shape == (experts, MOE_INTERMEDIATE, HIDDEN), down.shape
    for expert in range(experts):
        assert float(down[expert].unique().item()) == 100.0 + expert

    # Shared MLP down_proj: [hidden, intermediate] -> [intermediate, hidden].
    shared_down = torch.arange(HIDDEN * MOE_INTERMEDIATE, dtype=torch.float32).reshape(
        HIDDEN, MOE_INTERMEDIATE
    )
    assert torch.equal(_transpose_loader().load([FakeSlice(shared_down)], 0), shared_down.T)

    # Shared MLP: chunk, then shard the intermediate dim across ranks.
    marked = torch.cat(
        (
            torch.arange(MOE_INTERMEDIATE, dtype=torch.float32).unsqueeze(1).expand(-1, HIDDEN),
            torch.arange(MOE_INTERMEDIATE, dtype=torch.float32).unsqueeze(1).expand(-1, HIDDEN)
            + 1000.0,
        ),
        dim=0,
    ).contiguous()
    for tp_size in (16, 32):
        shard_size = MOE_INTERMEDIATE // tp_size
        for chunk, offset in ((0, 0.0), (1, 1000.0)):
            loader = _fused_chunk_sharded_loader(chunk, 2, shard_size, tp_size)
            for rank in range(tp_size):
                shard = loader.load([FakeSlice(marked)], rank)
                assert shard.shape == (HIDDEN, shard_size)
                expected = torch.arange(
                    rank * shard_size, (rank + 1) * shard_size, dtype=torch.float32
                ) + offset
                assert torch.equal(shard[0], expected), (
                    f"tp{tp_size} chunk{chunk} rank{rank}: {shard[0][:4].tolist()} != "
                    f"{expected[:4].tolist()}"
                )
    print("stacked expert loaders OK")


if __name__ == "__main__":
    test_interleaved_qkv_loader()
    test_o_proj_loader_matches_query_heads()
    test_stacked_expert_loaders()
    print("OK")
