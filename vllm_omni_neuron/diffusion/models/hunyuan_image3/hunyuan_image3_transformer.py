# SPDX-License-Identifier: Apache-2.0
"""NeuronHunyuanImage3Transformer — HunyuanImage-3.0 DiT backbone for Neuron.

Native PyTorch + ``torch.compile`` + ``libtorch-neuronx-lite`` + NKI, following the
Wan2.2 blueprint in ``vllm_omni_neuron/diffusion/models/wan2_2/wan2_2_transformer.py``:
raw ``nn.Parameter`` tensors with ``vllm_neuron.utils.weight_loader`` loaders (no
``ColumnParallelLinear`` / ``RowParallelLinear`` wrappers), NKI-Lib kernels behind a
``can_run_kernel`` gate with torch fallbacks, and the pure-math helpers (2D RoPE,
timestep / patch embedders) imported from upstream ``vllm_omni``.

Two graphs, both fixed-shape so a cold compile is paid once per deployment:

``forward_prefill(inputs_embeds, cos, sin)``
    Causal self-attention over the text prompt, right-padded to ``prefill_len``.
    Returns the per-layer prompt K/V. Runs once per request.

``forward_denoise(hidden, cos, sin, key_bias, *prompt_kv)``
    One denoising step over ``[timestep token] + image tokens``, attending to the
    cached prompt K/V plus the step's own image K/V. Replayed every scheduler step.

Splitting the first step's single forward into prefill + denoise is exact for this
model, not an approximation: HunyuanImage3's generation mask is lower-triangular with
a full-attention block over the generated-image span, so prompt tokens never attend
to image tokens, and image tokens attend to the whole prompt plus the whole image
block. ``key_bias`` masks only the prompt right-padding.

Parallelism (world == ``tensor_parallel_size``):
  * attention — head-parallel (``num_attention_heads`` // tp), GQA K/V replicated
    when ``num_key_value_heads < tp``, row-parallel o-proj + all-reduce;
  * routed MoE — expert-parallel (``num_experts`` // tp experts per rank), every rank
    evaluates the full router and contributes only its own experts' share;
  * shared MLP — intermediate-parallel, folded into the same all-reduce.
"""

import logging
import math
import os
from collections.abc import Iterable
from functools import cache

import nki
import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.mlp.mlp import mlp as nkilib_mlp
from nkilib.core.output_projection.output_projection_cte import output_projection_cte
from nkilib.core.utils.common_types import (
    ActFnType,
    DtypeMode,
    NormType,
    QuantizationType,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    get_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from vllm_omni_neuron import envs
from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups
from vllm_omni_neuron.lite_compat import nki_op

logger = logging.getLogger(__name__)


# ===================================================================
# NKI kernel helpers
# ===================================================================


def _resolve_can_run_kernel():
    """Resolve the kernel gate from the installed vLLM-Neuron stack."""
    try:
        from vllm_neuron.utils.neuron_utils import can_run_kernel
    except ModuleNotFoundError as error:
        if error.name != "vllm_neuron.utils.neuron_utils":
            raise
        from vllm_neuron.nki.nki_hop import can_run_kernel

    return can_run_kernel


@cache
def _can_run_kernel_impl():
    return _resolve_can_run_kernel()


def can_run_kernel(tensor) -> bool:
    """Run the import-safe kernel availability gate (CPU mode / fake tensors -> False)."""
    return _can_run_kernel_impl()(tensor)


SUPPORTED_LNC = (2,)


def coerce_lnc(lnc) -> int:
    """Validate a raw ``NEURON_LOGICAL_NC_CONFIG`` value and return it as a supported int."""
    if lnc is None:
        return 2
    if lnc not in SUPPORTED_LNC:
        raise ValueError(
            f"NEURON_LOGICAL_NC_CONFIG={lnc!r} is not supported; valid: {SUPPORTED_LNC}."
        )
    return lnc


def _wrap_nki_kernel(kernel):
    """Return a traceable Lite NKI HOP using the configured LNC launch."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(kernel)[coerce_lnc(envs.NEURON_LOGICAL_NC_CONFIG)]


# --- causal flash attention (prompt prefill) -----------------------------------


@nki.jit
def _hunyuan_causal_attention_kernel(q, k, v):
    """Causal flash attention with a **d-major** output (``tp_out=True``).

    Takes ``[BN, S, D]`` q/k/v and returns ``[BN, D, S]``, which is exactly the layout
    :func:`_hunyuan_o_proj` consumes, so the whole attention output never needs an HBM
    transpose. ``scale=1.0``: the caller pre-scales Q (see :func:`_prefill_attend`).
    """
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=True,
        tp_q=True,
        tp_k=True,
        tp_out=True,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


@nki_op("hunyuan_image3::causal_attention_cte")
def _hunyuan_nki_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return _wrap_nki_kernel(_hunyuan_causal_attention_kernel)(q, k, v)


# attention_cte tiling limits, mirrored from
# vllm_neuron.functional.attention.attention_cte._can_use_flash_attention_kernel.
_ATTN_MAX_BS = 512
_ATTN_MAX_SEQLEN = 131072
_ATTN_MAX_HEAD_DIM = 128


def _can_use_causal_attention_kernel(q, k, v) -> bool:
    """Whether ``attention_cte`` can run for these ``[B, N, S, D]`` operands."""
    if not can_run_kernel(v):
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    b, n, s_q, d = q.shape
    if b * n > _ATTN_MAX_BS:
        return False
    if s_q > _ATTN_MAX_SEQLEN or k.shape[2] > _ATTN_MAX_SEQLEN:
        return False
    return d <= _ATTN_MAX_HEAD_DIM


def _prefill_attend(query, key, value, scale):
    """Causal attention over ``[B, N, S, D]`` q/k/v, returning d-major ``[B, N, D, S]``.

    Q is pre-scaled so the kernel can run at ``scale=1.0``; the torch fallback folds the
    same scale into its float32 score matrix, and both softmax in float32.
    """
    scaled_query = query * scale
    if _can_use_causal_attention_kernel(scaled_query, key, value):
        b, n, s_q, d = scaled_query.shape
        out = _hunyuan_nki_causal_attention(
            scaled_query.reshape(b * n, s_q, d).contiguous(),
            key.reshape(b * n, key.shape[2], d).contiguous(),
            value.reshape(b * n, value.shape[2], d).contiguous(),
        )
        return out.reshape(b, n, d, s_q)

    seq = scaled_query.shape[2]
    causal = torch.ones(seq, key.shape[2], dtype=torch.bool, device=query.device).tril()
    scores = torch.matmul(scaled_query.float(), key.float().transpose(-2, -1))
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value.float()).to(query.dtype).transpose(-2, -1)


def _masked_attend(query, key, value, scale, key_bias):
    """Softmax attention with an additive ``[1, 1, 1, S_k]`` key bias, d-major output.

    Kept in torch (the Neuron compiler fuses it) because ``attention_cte`` takes no
    per-position mask, and because the denoise attention is ~1.5% of a layer's FLOPs
    next to the 64-expert MoE. Softmax runs in float32 for parity with the kernel's
    ``softmax_dtype=nl.float32`` used on the prefill side.
    """
    scores = torch.matmul(query.float() * scale, key.float().transpose(-2, -1))
    if key_bias is not None:
        scores = scores + key_bias
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value.float()).to(query.dtype).transpose(-2, -1)


def _denoise_attend(query, key, value, scale, key_biases):
    """Attention of ``[timestep token] + image tokens`` over ``[prompt K/V | image K/V]``.

    Two key biases, because the generation mask gives the two query kinds different
    visibility: image tokens see the whole prompt and the whole image block (the
    full-attention span), while the timestep token is only causal, so it sees the
    prompt and itself. Both are additive ``[1, 1, 1, S_k]`` vectors — the prompt's
    right-padding columns are masked in both — and the pipeline reads them straight out
    of upstream's generation mask, so this stays exact even if the span's boundary
    moves. The timestep row is computed separately rather than with a full
    ``[S_q, S_k]`` mask: one extra row of attention costs far less than materializing
    and adding a 4097 x 4609 mask in all 32 layers.
    """
    timestep_bias, image_bias = key_biases
    image_out = _masked_attend(query, key, value, scale, image_bias)
    timestep_out = _masked_attend(query[:, :, :1], key, value, scale, timestep_bias)
    return torch.cat((timestep_out, image_out[..., 1:]), dim=-1)


# --- output projection ---------------------------------------------------------


@nki.jit
def _hunyuan_o_proj_kernel(active, weight, bias):
    """Output projection: ``[B, N, D, S] x [N*D, H] (+[1, H]) -> [B, S, H]``."""
    return output_projection_cte(
        active,
        weight,
        bias,
        QuantizationType.NONE,
        None,
        None,
    )


@nki_op("hunyuan_image3::output_projection_cte")
def _hunyuan_nki_o_proj(
    active: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _wrap_nki_kernel(_hunyuan_o_proj_kernel)(active, weight, bias)


MAX_O_PROJ_HEADS = 17


def _can_use_o_proj_kernel(active, weight, bias) -> bool:
    if not can_run_kernel(active):
        return False
    if active.dim() != 4 or weight.dim() != 2:
        return False
    b, n, d, s = active.shape
    nd, h = weight.shape
    if bias.dim() != 2 or bias.shape != (1, h):
        return False
    return (
        n * d == nd
        and n <= MAX_O_PROJ_HEADS
        and d <= 128
        and h <= 16384 + 4321
        and b * s <= 128 * 1024
        and h % 2 == 0
    )


def _hunyuan_o_proj(active, weight, bias):
    """``[B, N, D, S]`` attention output -> ``[B, S, H]`` via NKI, with a torch fallback."""
    if _can_use_o_proj_kernel(active, weight, bias):
        return _hunyuan_nki_o_proj(active, weight, bias)
    b, n, d, s = active.shape
    x = active.reshape(b, n * d, s).transpose(1, 2)
    return torch.matmul(x, weight) + bias


# --- gated (SwiGLU) MLP -------------------------------------------------------


@nki.jit
def _hunyuan_swiglu_mlp_kernel(hidden, gate_weight, up_weight, down_weight):
    """``down(silu(hidden @ gate) * (hidden @ up))`` for ``hidden`` ``[B, T, H]``.

    HunyuanImage3 stores one fused ``gate_and_up_proj`` whose first chunk is the
    *linear* branch and whose second chunk is the *gated* branch
    (``down_proj(x1 * silu(x2))`` in the reference implementation), so the loader
    splits it into ``up_weight`` (chunk 0) and ``gate_weight`` (chunk 1) and this
    kernel's naming follows nkilib's ``act(gate) * up`` convention.
    """
    return nkilib_mlp(
        hidden_tensor=hidden,
        gate_proj_weights_tensor=gate_weight,
        up_proj_weights_tensor=up_weight,
        down_proj_weights_tensor=down_weight,
        normalization_weights_tensor=None,
        gate_proj_bias_tensor=None,
        up_proj_bias_tensor=None,
        down_proj_bias_tensor=None,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.SiLU,
        normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        gate_w_scale=None,
        up_w_scale=None,
        down_w_scale=None,
        gate_up_in_scale=None,
        down_in_scale=None,
        quant_clipping_bound=0.0,
        output_dtype=None,
        store_output_in_sbuf=False,
        eps=1e-6,
        skip_gate_proj=False,
        use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True,
        use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        force_cte_mode=False,
        dtype_mode=DtypeMode.AUTO,
    )


@nki_op("hunyuan_image3::swiglu_mlp")
def _hunyuan_nki_swiglu_mlp(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    return _wrap_nki_kernel(_hunyuan_swiglu_mlp_kernel)(
        hidden, gate_weight, up_weight, down_weight
    )


# nkilib MLP tiling limits for the grid=2 (LNC=2) launch, mirrored from
# vllm_neuron.functional.mlp._can_use_kernel. Both projections are bias-free here.
_MLP_TKG_BS_SEQLEN_THRESHOLD = 128
_MLP_SRC_PROJ_INT_DIM_TILE_SIZE = 512
_MLP_NUM_HW_PSUM_BANKS = 8


def _can_use_swiglu_mlp_kernel(hidden, gate_weight) -> bool:
    """Whether the nkilib MLP kernel can run for these ``[B, T, H]`` / ``[H, I]`` shapes."""
    if not can_run_kernel(hidden):
        return False
    if hidden.dim() != 3 or gate_weight.dim() != 2:
        return False
    b, t, h = hidden.shape
    inner_dim = gate_weight.shape[1]
    if h % 128 != 0:
        return False
    if b * t <= _MLP_TKG_BS_SEQLEN_THRESHOLD:
        # TKG: the hidden dim is sharded across 2 cores, so H // 128 must be even.
        return h % 256 == 0
    return math.ceil(inner_dim / _MLP_SRC_PROJ_INT_DIM_TILE_SIZE) <= _MLP_NUM_HW_PSUM_BANKS


def _hunyuan_swiglu_mlp(hidden, gate_weight, up_weight, down_weight, use_kernel: bool):
    """SwiGLU MLP via the NKI kernel when allowed, else the equivalent torch math."""
    if use_kernel and _can_use_swiglu_mlp_kernel(hidden, gate_weight):
        return _hunyuan_nki_swiglu_mlp(hidden, gate_weight, up_weight, down_weight)
    gated = F.silu(torch.matmul(hidden, gate_weight)) * torch.matmul(hidden, up_weight)
    return torch.matmul(gated, down_weight)


# ===================================================================
# Weight loaders
# ===================================================================


def _cast_weight_loader(loader, dtype, name: str, logged_casts: set):
    """Cast a loader's output to the parameter dtype, logging each cast kind once."""
    weight_type = ".".join("*" if part.isdigit() else part for part in name.split("."))

    def transform(slices, rank):
        tensor = loader.load(slices, rank)
        cast_key = (weight_type, tensor.dtype, dtype)
        if tensor.dtype != dtype and cast_key not in logged_casts:
            logged_casts.add(cast_key)
            logger.info(
                "Casting weight %s from checkpoint dtype %s to parameter dtype %s", *cast_key
            )
        return tensor.to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def _interleaved_qkv_loader(
    num_kv_heads: int,
    num_kv_groups: int,
    head_dim: int,
    tp_size: int,
    num_heads_per_rank: int,
    num_kv_heads_per_rank: int,
) -> SafetensorsWeightLoader:
    """Shard HunyuanImage3's fused ``qkv_proj`` into a transposed per-rank ``[H, qkv]``.

    The checkpoint stores one ``[(kv_heads * (groups + 2)) * head_dim, hidden]`` tensor
    whose rows are grouped **per KV head**: ``groups`` query heads, then that group's K
    head, then its V head (see ``HunyuanImage3Model._split_qkv_weight`` upstream). This
    loader reshapes to ``[kv_heads, groups + 2, head_dim, hidden]``, takes this rank's
    query-head and KV-head slices, and returns ``[hidden, q + k + v]`` so the forward is
    a plain ``x @ W``.

    When ``num_key_value_heads < tp_size`` the KV heads are *replicated*: ranks sharing a
    KV head each load the whole head, matching vLLM's ``max(1, total_kv // tp)`` rule.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        weight = slices[0][:]
        hidden = weight.shape[1]
        weight = weight.reshape(num_kv_heads, num_kv_groups + 2, head_dim, hidden)
        q_all = weight[:, :num_kv_groups].reshape(-1, head_dim, hidden)
        k_all = weight[:, num_kv_groups].reshape(-1, head_dim, hidden)
        v_all = weight[:, num_kv_groups + 1].reshape(-1, head_dim, hidden)

        q_start = rank * num_heads_per_rank
        q = q_all[q_start : q_start + num_heads_per_rank].reshape(-1, hidden)

        if num_kv_heads >= tp_size:
            kv_start = rank * num_kv_heads_per_rank
        else:
            # Replicated: ranks are grouped so that each KV head serves tp // kv_heads ranks.
            kv_start = (rank * num_kv_heads) // tp_size
        kv_stop = kv_start + num_kv_heads_per_rank
        k = k_all[kv_start:kv_stop].reshape(-1, hidden)
        v = v_all[kv_start:kv_stop].reshape(-1, hidden)

        return torch.cat((q, k, v), dim=0).T

    return SafetensorsWeightLoader(transform=transform)


def _fused_chunk_loader(chunk: int, num_chunks: int) -> SafetensorsWeightLoader:
    """Take one chunk of a fused ``[num_chunks * I, H]`` weight, transposed to ``[H, I]``."""

    def transform(slices, rank):
        assert len(slices) == 1
        total = slices[0].get_shape()[0]
        size = total // num_chunks
        start = chunk * size
        return slices[0][start : start + size, :].T

    return SafetensorsWeightLoader(transform=transform)


def _fused_chunk_sharded_loader(
    chunk: int, num_chunks: int, shard_size: int, num_shards: int
) -> SafetensorsWeightLoader:
    """Chunk a fused ``[num_chunks * I, H]`` weight, then shard ``I`` -> ``[H, I/tp]``."""

    def transform(slices, rank):
        assert len(slices) == 1
        total = slices[0].get_shape()[0]
        size = total // num_chunks
        start = chunk * size + (rank % num_shards) * shard_size
        return slices[0][start : start + shard_size, :].T

    return SafetensorsWeightLoader(transform=transform)


def _transpose_loader() -> SafetensorsWeightLoader:
    """Load a ``[out, in]`` checkpoint weight as ``[in, out]``."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].T)


def _identity_loader() -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader()


# ===================================================================
# Modules
# ===================================================================


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HunyuanRMSNorm: float32 statistics, cast back, then scale."""
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return weight * xf.to(input_dtype)


def apply_rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX-style RoPE for ``x`` ``[B, S, N, D]`` with half-width ``cos``/``sin`` ``[B, S, D/2]``.

    Mirrors ``vllm_omni.diffusion.layers.rope.apply_rotary_emb_torch`` (non-interleaved):
    ``cos``/``sin`` are tiled to full width and the rotation splits ``D`` in halves. The
    rotation runs in float32 like upstream's ``HunYuanRotary2DEmbedder``, then casts back.
    """
    dtype = x.dtype
    xf = x.to(torch.float32)
    cos_f = torch.cat((cos, cos), dim=-1).unsqueeze(2)
    sin_f = torch.cat((sin, sin), dim=-1).unsqueeze(2)
    half = xf.shape[-1] // 2
    rotated = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
    return (xf * cos_f + rotated * sin_f).to(dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[B, S, N_kv, D]`` -> ``[B, S, N_kv * n_rep, D]`` (interleaved per KV head)."""
    if n_rep == 1:
        return x
    b, s, n_kv, d = x.shape
    return x[:, :, :, None, :].expand(b, s, n_kv, n_rep, d).reshape(b, s, n_kv * n_rep, d)


class NeuronHunyuanAttention(nn.Module):
    """Head-parallel self-attention with fused QKV, 2D RoPE and per-head QK RMSNorm.

    RoPE is applied *before* the QK norm, matching both the reference implementation and
    upstream ``HunYuanAttention``.
    """

    def __init__(self, config, tp_size: int, tp_group):
        super().__init__()
        self.head_dim = int(config.attention_head_dim)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        )
        self.tp_size = tp_size
        self.tp_group = tp_group

        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} must be divisible by "
                f"tensor_parallel_size={tp_size}"
            )
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads={self.total_num_kv_heads} must be divisible by "
                    f"tensor_parallel_size={tp_size}"
                )
        elif tp_size % self.total_num_kv_heads != 0:
            raise ValueError(
                f"tensor_parallel_size={tp_size} must be a multiple of "
                f"num_key_value_heads={self.total_num_kv_heads} so KV heads replicate evenly"
            )

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_split = [self.q_size, self.q_size + self.kv_size]
        self.scale = self.head_dim**-0.5
        self.eps = float(config.rms_norm_eps)
        self.use_qk_norm = bool(getattr(config, "use_qk_norm", False))

        hidden_size = int(config.hidden_size)
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(hidden_size, self.q_size + 2 * self.kv_size)
        )
        self.o_proj_weight = nn.Parameter(torch.empty(self.q_size, hidden_size))
        if self.use_qk_norm:
            self.query_layernorm_weight = nn.Parameter(torch.empty(self.head_dim))
            self.key_layernorm_weight = nn.Parameter(torch.empty(self.head_dim))
        # output_projection_cte always takes a bias; HunyuanImage3 has none
        # (attention_bias=False), so carry a constant zero rather than branching.
        self.register_buffer(
            "o_proj_zero_bias", torch.zeros(1, hidden_size), persistent=False
        )

    def _project(self, hidden_states, cos, sin):
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)
        q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
        bsz, seq = hidden_states.shape[0], hidden_states.shape[1]
        query = q.reshape(bsz, seq, self.num_heads, self.head_dim)
        key = k.reshape(bsz, seq, self.num_kv_heads, self.head_dim)
        value = v.reshape(bsz, seq, self.num_kv_heads, self.head_dim)

        query = apply_rope_2d(query, cos, sin)
        key = apply_rope_2d(key, cos, sin)
        if self.use_qk_norm:
            query = rms_norm(query, self.query_layernorm_weight, self.eps)
            key = rms_norm(key, self.key_layernorm_weight, self.eps)
        return query, key, value

    def _finish(self, attention_dmajor):
        out = _hunyuan_o_proj(
            attention_dmajor,
            self.o_proj_weight,
            self.o_proj_zero_bias,
        )
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        return out

    def forward_prefill(self, hidden_states, cos, sin):
        """Causal prefill; also returns this rank's prompt K/V (pre-GQA-expansion)."""
        query, key, value = self._project(hidden_states, cos, sin)
        attn = _prefill_attend(
            query.transpose(1, 2),
            repeat_kv(key, self.num_kv_groups).transpose(1, 2),
            repeat_kv(value, self.num_kv_groups).transpose(1, 2),
            self.scale,
        )
        return self._finish(attn), key, value

    def forward_denoise(self, hidden_states, cos, sin, key_biases, prompt_key, prompt_value):
        query, key, value = self._project(hidden_states, cos, sin)
        key = torch.cat((prompt_key, key), dim=1)
        value = torch.cat((prompt_value, value), dim=1)
        attn = _denoise_attend(
            query.transpose(1, 2),
            repeat_kv(key, self.num_kv_groups).transpose(1, 2),
            repeat_kv(value, self.num_kv_groups).transpose(1, 2),
            self.scale,
            key_biases,
        )
        return self._finish(attn)


class NeuronHunyuanMoE(nn.Module):
    """Expert-parallel routed MoE plus an intermediate-parallel shared MLP.

    Every rank evaluates the full float32 router (as the reference implementation does:
    ``softmax`` -> top-k -> renormalise) and then contributes only the experts it owns;
    the final all-reduce sums the shards. Top-k selection is expressed as a threshold on
    the k-th largest probability instead of a scatter so the whole block stays inside one
    static graph.
    """

    def __init__(self, config, layer_idx: int, tp_size: int, tp_rank: int, tp_group, use_kernel: bool):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.use_kernel = use_kernel

        num_experts = config.num_experts
        self.num_experts = int(
            num_experts if isinstance(num_experts, int) else num_experts[layer_idx]
        )
        topk = config.moe_topk
        self.top_k = int(topk if isinstance(topk, int) else topk[layer_idx])
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))

        moe_inter = getattr(config, "moe_intermediate_size", None) or config.intermediate_size
        self.moe_intermediate_size = int(
            moe_inter if isinstance(moe_inter, int) else moe_inter[layer_idx]
        )

        if self.num_experts % tp_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by "
                f"tensor_parallel_size={tp_size} for expert parallelism"
            )
        self.num_local_experts = self.num_experts // tp_size
        self.local_expert_start = tp_rank * self.num_local_experts

        # Router. The reference keeps ``wg`` in float32, so the parameter is float32 even
        # though the checkpoint stores bf16 values.
        self.gate_weight = nn.Parameter(
            torch.empty(hidden_size, self.num_experts, dtype=torch.float32)
        )

        inter = self.moe_intermediate_size
        self.expert_gate_weight = nn.ParameterList(
            nn.Parameter(torch.empty(hidden_size, inter)) for _ in range(self.num_local_experts)
        )
        self.expert_up_weight = nn.ParameterList(
            nn.Parameter(torch.empty(hidden_size, inter)) for _ in range(self.num_local_experts)
        )
        self.expert_down_weight = nn.ParameterList(
            nn.Parameter(torch.empty(inter, hidden_size)) for _ in range(self.num_local_experts)
        )

        num_shared = getattr(config, "num_shared_expert", 0)
        if isinstance(num_shared, list):
            num_shared = num_shared[layer_idx]
        self.num_shared_expert = int(num_shared) if getattr(config, "use_mixed_mlp_moe", 0) else 0
        if self.num_shared_expert:
            shared_inter = int(config.intermediate_size) * self.num_shared_expert
            if shared_inter % tp_size != 0:
                raise ValueError(
                    f"shared MLP intermediate size {shared_inter} must be divisible by "
                    f"tensor_parallel_size={tp_size}"
                )
            self.shared_intermediate_size = shared_inter // tp_size
            self.shared_gate_weight = nn.Parameter(
                torch.empty(hidden_size, self.shared_intermediate_size)
            )
            self.shared_up_weight = nn.Parameter(
                torch.empty(hidden_size, self.shared_intermediate_size)
            )
            self.shared_down_weight = nn.Parameter(
                torch.empty(self.shared_intermediate_size, hidden_size)
            )

    def _routing_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Dense ``[T, num_local_experts]`` routing weights for this rank's experts."""
        logits = torch.matmul(tokens.to(torch.float32), self.gate_weight)
        probs = torch.softmax(logits, dim=-1)
        threshold = torch.topk(probs, self.top_k, dim=-1).values[..., -1:]
        dense = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        if self.norm_topk_prob and self.top_k > 1:
            dense = dense / dense.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return dense[
            :, self.local_expert_start : self.local_expert_start + self.num_local_experts
        ]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq, hidden = hidden_states.shape
        tokens = hidden_states.reshape(1, bsz * seq, hidden)
        weights = self._routing_weights(tokens[0]).to(hidden_states.dtype)

        output = None
        for expert in range(self.num_local_experts):
            expert_out = _hunyuan_swiglu_mlp(
                tokens,
                self.expert_gate_weight[expert],
                self.expert_up_weight[expert],
                self.expert_down_weight[expert],
                self.use_kernel,
            )
            scaled = expert_out * weights[:, expert].reshape(1, -1, 1)
            output = scaled if output is None else output + scaled

        if self.num_shared_expert:
            shared = _hunyuan_swiglu_mlp(
                tokens,
                self.shared_gate_weight,
                self.shared_up_weight,
                self.shared_down_weight,
                self.use_kernel,
            )
            output = shared if output is None else output + shared

        if self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output.reshape(bsz, seq, hidden)


class NeuronHunyuanDecoderLayer(nn.Module):
    """Pre-norm decoder layer: RMSNorm -> attention -> residual -> RMSNorm -> MoE."""

    def __init__(self, config, layer_idx: int, tp_size: int, tp_rank: int, tp_group, use_kernel: bool):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.eps = float(config.rms_norm_eps)
        self.input_layernorm_weight = nn.Parameter(torch.empty(hidden_size))
        self.post_attention_layernorm_weight = nn.Parameter(torch.empty(hidden_size))
        self.self_attn = NeuronHunyuanAttention(config, tp_size, tp_group)
        self.mlp = NeuronHunyuanMoE(config, layer_idx, tp_size, tp_rank, tp_group, use_kernel)

    def forward_prefill(self, hidden_states, cos, sin):
        residual = hidden_states
        normed = rms_norm(hidden_states, self.input_layernorm_weight, self.eps)
        attn_out, key, value = self.self_attn.forward_prefill(normed, cos, sin)
        hidden_states = residual + attn_out
        residual = hidden_states
        normed = rms_norm(hidden_states, self.post_attention_layernorm_weight, self.eps)
        hidden_states = residual + self.mlp(normed)
        return hidden_states, key, value

    def forward_denoise(self, hidden_states, cos, sin, key_biases, prompt_key, prompt_value):
        residual = hidden_states
        normed = rms_norm(hidden_states, self.input_layernorm_weight, self.eps)
        attn_out = self.self_attn.forward_denoise(
            normed, cos, sin, key_biases, prompt_key, prompt_value
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        normed = rms_norm(hidden_states, self.post_attention_layernorm_weight, self.eps)
        return residual + self.mlp(normed)


class NeuronHunyuanImage3Transformer(nn.Module):
    """The 32-layer HunyuanImage-3.0 MoE backbone, sharded for Neuron.

    Only the decoder stack lives here. The token embedding, timestep embedders, patch
    embedder, final layer and VAE stay with the pipeline (see
    ``pipeline_hunyuan_image3.py``), which also owns the prompt padding and the
    ``key_bias`` that masks it.
    """

    def __init__(self, config, use_nki_mlp: bool = True):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group().device_group if self.tp_size > 1 else None

        # Make the TP group resolvable to its full replica-group partition so the
        # torch-native Neuron backend can legalize the all-reduces for SPMD
        # compilation (otherwise: "replica id #N not seen in replica groups").
        register_replica_groups(tp_size=self.tp_size, cp_size=1)

        self.num_layers = int(config.num_hidden_layers)
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(config.attention_head_dim)
        self.layers = nn.ModuleList(
            NeuronHunyuanDecoderLayer(
                config,
                layer_idx,
                self.tp_size,
                self.tp_rank,
                self.tp_group,
                use_nki_mlp,
            )
            for layer_idx in range(self.num_layers)
        )
        self._compiled_prefill = None

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0].input_layernorm_weight.dtype

    # ---- graphs -------------------------------------------------------------

    def forward_prefill(self, inputs_embeds, cos, sin):
        """Prompt prefill. Returns ``2 * num_layers`` K/V tensors, layer-major."""
        hidden_states = inputs_embeds
        kv: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, key, value = layer.forward_prefill(hidden_states, cos, sin)
            kv.append(key)
            kv.append(value)
        return tuple(kv)

    def forward_denoise(self, hidden_states, cos, sin, key_biases, *prompt_kv):
        """One denoising step over ``[timestep token] + image tokens``."""
        if len(prompt_kv) != 2 * self.num_layers:
            raise ValueError(
                f"expected {2 * self.num_layers} prompt K/V tensors, got {len(prompt_kv)}"
            )
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_denoise(
                hidden_states,
                cos,
                sin,
                key_biases,
                prompt_kv[2 * layer_idx],
                prompt_kv[2 * layer_idx + 1],
            )
        return hidden_states

    def forward(self, *args, **kwargs):  # pragma: no cover - explicit entry points only
        raise RuntimeError(
            "Use forward_prefill()/forward_denoise(); this backbone has two graphs."
        )

    # ---- compilation --------------------------------------------------------

    def compile(self, *args, options=None, **kwargs):
        """Compile the prefill graph.

        ``forward_denoise`` is deliberately *not* compiled here: the pipeline traces it
        as part of one fused denoise-step graph (patch embed -> backbone -> final
        layer), so compiling it separately would emit a second, never-replayed NEFF.
        """
        options = dict(options or {})
        kwargs.setdefault("fullgraph", True)
        kwargs.setdefault("dynamic", False)
        self._compiled_prefill = torch.compile(
            self.forward_prefill,
            *args,
            options={**options, "model_name": "hunyuan_image3_prefill"},
            **kwargs,
        )
        return self

    def run_prefill(self, inputs_embeds, cos, sin):
        fn = self._compiled_prefill or self.forward_prefill
        return fn(inputs_embeds, cos, sin)

    # ---- weights ------------------------------------------------------------

    def checkpoint_mappings(self) -> dict[str, str | list[str]]:
        """``{parameter name -> checkpoint key}`` for the decoder stack.

        Checkpoint keys are the raw HunyuanImage-3.0 names (``model.layers.N....``);
        every parameter is listed because ``load_sharded_pipelined`` falls back to the
        parameter's own name for anything missing.
        """
        mappings: dict[str, str | list[str]] = {}
        for i in range(self.num_layers):
            src = f"model.layers.{i}"
            mappings[f"layers.{i}.input_layernorm_weight"] = f"{src}.input_layernorm.weight"
            mappings[f"layers.{i}.post_attention_layernorm_weight"] = (
                f"{src}.post_attention_layernorm.weight"
            )
            mappings[f"layers.{i}.self_attn.qkv_proj_weight"] = f"{src}.self_attn.qkv_proj.weight"
            mappings[f"layers.{i}.self_attn.o_proj_weight"] = f"{src}.self_attn.o_proj.weight"
            mappings[f"layers.{i}.self_attn.query_layernorm_weight"] = (
                f"{src}.self_attn.query_layernorm.weight"
            )
            mappings[f"layers.{i}.self_attn.key_layernorm_weight"] = (
                f"{src}.self_attn.key_layernorm.weight"
            )
            mappings[f"layers.{i}.mlp.gate_weight"] = f"{src}.mlp.gate.wg.weight"
            moe = self.layers[i].mlp
            for local, expert in enumerate(
                range(moe.local_expert_start, moe.local_expert_start + moe.num_local_experts)
            ):
                fused = f"{src}.mlp.experts.{expert}.gate_and_up_proj.weight"
                mappings[f"layers.{i}.mlp.expert_up_weight.{local}"] = fused
                mappings[f"layers.{i}.mlp.expert_gate_weight.{local}"] = fused
                mappings[f"layers.{i}.mlp.expert_down_weight.{local}"] = (
                    f"{src}.mlp.experts.{expert}.down_proj.weight"
                )
            if moe.num_shared_expert:
                fused = f"{src}.mlp.shared_mlp.gate_and_up_proj.weight"
                mappings[f"layers.{i}.mlp.shared_up_weight"] = fused
                mappings[f"layers.{i}.mlp.shared_gate_weight"] = fused
                mappings[f"layers.{i}.mlp.shared_down_weight"] = (
                    f"{src}.mlp.shared_mlp.down_proj.weight"
                )
        return mappings

    def _attach_weight_loaders(self) -> None:
        """Attach the per-parameter sharding / fusion loaders."""
        tp_size = self.tp_size
        for i in range(self.num_layers):
            layer = self.layers[i]
            attn = layer.self_attn
            set_weight_loader(layer.input_layernorm_weight, _identity_loader())
            set_weight_loader(layer.post_attention_layernorm_weight, _identity_loader())
            set_weight_loader(
                attn.qkv_proj_weight,
                _interleaved_qkv_loader(
                    num_kv_heads=attn.total_num_kv_heads,
                    num_kv_groups=attn.total_num_heads // attn.total_num_kv_heads,
                    head_dim=attn.head_dim,
                    tp_size=tp_size,
                    num_heads_per_rank=attn.num_heads,
                    num_kv_heads_per_rank=attn.num_kv_heads,
                ),
            )
            # Row-parallel o-proj: shard the input dim of a [out, in] checkpoint weight.
            set_weight_loader(
                attn.o_proj_weight,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=attn.q_size,
                    num_shards=tp_size,
                    is_storage_transposed=True,
                ),
            )
            if attn.use_qk_norm:
                set_weight_loader(attn.query_layernorm_weight, _identity_loader())
                set_weight_loader(attn.key_layernorm_weight, _identity_loader())

            moe = layer.mlp
            set_weight_loader(moe.gate_weight, _transpose_loader())
            for local in range(moe.num_local_experts):
                # ``gate_and_up_proj`` is [up | gate]: the reference computes
                # down(x1 * silu(x2)) over chunk 0 and chunk 1 respectively.
                set_weight_loader(moe.expert_up_weight[local], _fused_chunk_loader(0, 2))
                set_weight_loader(moe.expert_gate_weight[local], _fused_chunk_loader(1, 2))
                set_weight_loader(moe.expert_down_weight[local], _transpose_loader())
            if moe.num_shared_expert:
                set_weight_loader(
                    moe.shared_up_weight,
                    _fused_chunk_sharded_loader(0, 2, moe.shared_intermediate_size, tp_size),
                )
                set_weight_loader(
                    moe.shared_gate_weight,
                    _fused_chunk_sharded_loader(1, 2, moe.shared_intermediate_size, tp_size),
                )
                set_weight_loader(
                    moe.shared_down_weight,
                    sharding_weight_loader(
                        shard_dim=0,
                        shard_size=moe.shared_intermediate_size,
                        num_shards=tp_size,
                        is_storage_transposed=True,
                    ),
                )

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        cache_dir: str | None = None,
    ) -> None:
        """Load this rank's shard of the decoder stack with pipelined data movement."""
        self._attach_weight_loaders()
        mappings = self.checkpoint_mappings()

        logged_casts: set = set()
        for name, param in self.named_parameters():
            set_weight_loader(
                param,
                _cast_weight_loader(get_weight_loader(param), param.dtype, name, logged_casts),
            )

        checkpoint = SafetensorsCheckpoint(model_name_or_path, cache_dir)
        result = checkpoint.load_sharded_pipelined(
            self.tp_rank,
            self.tp_size,
            self,
            mappings,
            device,
        )
        self.load_state_dict(result.state_dict, strict=False, assign=True)


def expected_checkpoint_keys(transformer: NeuronHunyuanImage3Transformer) -> Iterable[str]:
    """Checkpoint keys the backbone consumes — used by the pipeline's loader split."""
    for value in transformer.checkpoint_mappings().values():
        if isinstance(value, list):
            yield from value
        else:
            yield value


def nki_mlp_enabled(model_config: dict | None) -> bool:
    """Resolve the ``moe_kernel`` stage-config knob (``nki`` default, ``torch`` opt-out)."""
    choice = str((model_config or {}).get("moe_kernel", "nki")).lower()
    if choice not in ("nki", "torch"):
        raise ValueError(f"model_config.moe_kernel must be 'nki' or 'torch', got {choice!r}")
    if choice == "torch":
        logger.warning("HunyuanImage3 MoE: NKI MLP kernel disabled by model_config.moe_kernel")
    return choice == "nki"


def default_prefill_len() -> int:
    """Default prompt bucket, overridable with ``model_config.prefill_len``."""
    return int(os.environ.get("VLLM_NEURON_HUNYUAN_PREFILL_LEN", "512"))
