# SPDX-License-Identifier: Apache-2.0
"""NeuronHunyuanImage3Pipeline — HunyuanImage-3.0 text-to-image for Neuron.

Subclasses vLLM-Omni's ``HunyuanImage3Pipeline`` and keeps all of its pure-Python
request handling — chat templating, 2D-RoPE construction, the generation attention
mask, image-info/resolution bucketing — while replacing everything from ``forward()``
down with the Neuron execution path:

* the 32-layer MoE backbone becomes :class:`NeuronHunyuanImage3Transformer`
  (head-parallel attention + expert-parallel MoE, NKI kernels, raw ``nn.Parameter``
  weights);
* the DiT runs as **two** fixed-shape ``torch.compile`` graphs — a prompt prefill that
  emits the per-layer prompt K/V, and a denoise step that fuses the patch embedder, the
  backbone and the final layer into one NEFF replayed every scheduler step;
* the scheduler, the CFG combine and the token embedding stay on the host, so no
  uncompiled op is ever dispatched to the device under the Lite runtime;
* the VAE decode runs on the host in float32 by default, or with ``vae_device: neuron``
  on the output rank's core as seven staged float16 graphs (:mod:`.vae_device`).

This is the DiT-only deployment: the prompt goes straight to the diffusion stage with
no autoregressive recaption stage, matching upstream's
``vllm_omni/deploy/hunyuan_image3_dit.yaml``.
"""

import json
import logging
import os
import time
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig
from vllm.transformers_utils.config import get_config
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import (
    TokenizerWrapper,
)
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3ImageProcessor,
    HunyuanImage3PreTrainedModel,
    ResBlock,
    TimestepEmbedder,
    UNetDown,
    UNetUp,
    real_batched_index_select,
    retrieve_timesteps,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
    get_hunyuan_image_3_pre_process_func,  # noqa: F401 - re-exported for the registry
)

from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    NeuronHunyuanImage3Transformer,
    cp_gather_batch,
    cp_gather_tokens,
    cp_local_tokens,
    default_prefill_len,
    denoise_attention_kernel_enabled,
    nki_attention_enabled,
    nki_mlp_enabled,
)
from vllm_omni_neuron.diffusion.models.hunyuan_image3.unet_parallel import (
    UNetShard,
    unet_down_tokens,
    unet_up_image,
)
from vllm_omni_neuron.lite_compat import is_lite_runtime

logger = logging.getLogger(__name__)


PIPELINE_REGISTRY = [
    {
        "model_arch": "HunyuanImage3ForCausalMM",
        "class_name": "NeuronHunyuanImage3Pipeline",
        "pre_process_func_name": "get_hunyuan_image_3_pre_process_func",
    },
]


# Additive bias for masked attention columns. Finite, so a fully masked row (which this
# model never produces) degrades to a uniform average instead of a NaN.
_MASK_BIAS = -1.0e30

# Modules replicated on every rank, loaded from the checkpoint by name.
_REPLICATED_PREFIXES = (
    "time_embed",
    "time_embed_2",
    "timestep_emb",
    "patch_embed",
    "final_layer",
)


# ===================================================================
# Patch embedder / final layer, traced without einops
# ===================================================================


def _group_norm_f32(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """GroupNorm in float32, returning float32.

    The reference implementation runs these UNet blocks under
    ``torch.autocast(bfloat16)``, which keeps normalisation on its float32 list and
    casts back down only at the next convolution. Returning float32 here (and casting
    at the conv call sites in :func:`_res_block`) reproduces that, so the adaptive-norm
    modulation and the SiLU after it keep full precision as they do on GPU.
    """
    return F.group_norm(
        x.float(), norm.num_groups, norm.weight.float(), norm.bias.float(), norm.eps
    )


def _res_block(block: ResBlock, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """``ResBlock.forward`` with float32 GroupNorms and no dropout (inference only)."""
    if block.updown:
        raise NotImplementedError(
            "HunyuanImage3 Neuron path expects patch_size=1 ResBlocks (no up/down sampling)"
        )
    conv_dtype = block.in_layers[2].weight.dtype
    h = F.silu(_group_norm_f32(block.in_layers[0], x))
    h = block.in_layers[2](h.to(conv_dtype))

    emb_out = block.emb_layers(emb)
    while emb_out.dim() < h.dim():
        emb_out = emb_out[..., None]
    scale, shift = torch.chunk(emb_out.float(), 2, dim=1)

    h = _group_norm_f32(block.out_layers[0], h) * (1.0 + scale) + shift
    h = F.silu(h)
    h = block.out_layers[3](h.to(conv_dtype))
    return block.skip_connection(x) + h


def _apply_unet_module(module: nn.Module, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """Dispatch one entry of a ``UNetDown``/``UNetUp`` ``ModuleList``."""
    if isinstance(module, ResBlock):
        return _res_block(module, x, emb)
    if isinstance(module, nn.Sequential):
        # UNetUp's out_norm tail: [GroupNorm, SiLU, Conv2d].
        x = F.silu(_group_norm_f32(module[0], x))
        return module[2](x.to(module[2].weight.dtype))
    return module(x)


def _unet_down_tokens(patch_embed: UNetDown, latents: torch.Tensor, t_emb: torch.Tensor):
    """``UNetDown.forward`` without einops, so the graph traces under ``fullgraph=True``."""
    x = latents
    for module in patch_embed.model:
        x = _apply_unet_module(module, x, t_emb)
    bsz, channels, token_h, token_w = x.shape
    tokens = x.reshape(bsz, channels, token_h * token_w).transpose(1, 2)
    return tokens, token_h, token_w


def _unet_up_image(
    final_layer: UNetUp,
    tokens: torch.Tensor,
    t_emb: torch.Tensor,
    token_h: int,
    token_w: int,
) -> torch.Tensor:
    """``UNetUp.forward`` without einops (see :func:`_unet_down_tokens`)."""
    bsz, _, channels = tokens.shape
    x = tokens.transpose(1, 2).reshape(bsz, channels, token_h, token_w)
    for module in final_layer.model:
        x = _apply_unet_module(module, x, t_emb)
    return x


# ===================================================================
# Pipeline
# ===================================================================


class NeuronHunyuanImage3Pipeline(HunyuanImage3Pipeline):
    """HunyuanImage-3.0-Instruct DiT text-to-image pipeline for AWS Trainium."""

    # Step execution assumes the upstream GPU denoise loop; the Neuron path owns its own
    # loop in forward(), so the engine drives this pipeline per request.
    supports_step_execution: ClassVar[bool] = False
    supports_request_batch: ClassVar[bool] = False
    # Conditioning images (image editing / IT2I). In vLLM-Omni 0.24 the flag only adds an
    # image to the engine's warmup request, which forward() skips.
    support_image_input: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["model"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = []

    def __init__(self, od_config, prefix: str = "") -> None:
        del prefix
        self.od_config = od_config
        # Config, generation config, tokenizer and weights all come from the one
        # snapshot of od_config.revision.
        model_path = self._resolve_model_path()
        self.hf_config = get_config(model_path, trust_remote_code=True)
        # PreTrainedModel.__init__ only records config/attributes — no device work.
        HunyuanImage3PreTrainedModel.__init__(self, self.hf_config)
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        # PreTrainedModel exposes `device` as a read-only property derived from the
        # parameters; this pipeline spans two devices (backbone on Neuron, embedding
        # and VAE on host), so pin it to this rank's core explicitly.
        self._device = get_local_device()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=True,
            )
        ]

        config = self.hf_config
        model_config = dict(od_config.model_config or {})
        # The request's prompt bucket, picked per prompt from _prefill_buckets.
        self.prefill_len = int(model_config.get("prefill_len", default_prefill_len()))
        self._prefill_buckets = _prompt_buckets(model_config)
        self.vae_dtype = getattr(torch, str(model_config.get("vae_dtype", "float32")))
        # Where the output rank decodes. `cpu` is the float32 reference; `neuron` puts the
        # decoder on this rank's core in 16 bits (vae_device_dtype) with fp32 reductions.
        self.vae_device = str(model_config.get("vae_device", "cpu"))
        if self.vae_device not in ("cpu", "neuron"):
            raise ValueError(f"vae_device must be 'cpu' or 'neuron', got {self.vae_device!r}")
        use_nki_mlp = nki_mlp_enabled(model_config)
        use_nki_attention = nki_attention_enabled(model_config)
        use_denoise_kernel = use_nki_attention and denoise_attention_kernel_enabled(model_config)
        moe_block_size = int(model_config.get("moe_block_size", 512))
        moe_implementation = str(model_config.get("moe_implementation", "shard_on_i_hybrid"))
        # Parallelism comes from parallel_config (tensor_parallel_size, and either
        # cfg_parallel_size: 2 or ring_degree), not from model_config.
        moved = {
            "context_parallel": "parallel_config.ring_degree",
            "cfg_parallel": "parallel_config.cfg_parallel_size",
        }
        for key in moved:
            if key in model_config:
                raise ValueError(f"model_config.{key} is gone: set {moved[key]} in the stage config")
        # CP self-attention: the plugin's tri-state. The ring kernel is MHA-only and has
        # no key mask or prefix, so it cannot express the denoise step's attention over
        # the right-padded prompt K/V plus the step's tokens: unset falls back to the
        # all-gather K/V path, true raises.
        if model_config.get("enable_ring_attention") is True:
            raise NotImplementedError(
                "enable_ring_attention: true -- the ring attention kernel cannot express "
                "HunyuanImage3's denoise attention (GQA, a padded prompt prefix, per-key "
                "biases); leave it unset or false for the all-gather K/V path"
            )
        # Split the patch embed and final layer by GroupNorm group across the TP ranks
        # (unet_parallel.py) instead of running them whole on every rank.
        self.unet_parallel = bool(model_config.get("unet_parallel", True))
        self._unet_shard = None
        # Cache-DiT DBCache over the denoise loop (_StepCache). Off by default: it trades
        # exactness for skipping most of the backbone on steps whose first layer barely
        # changes. True takes Cache-DiT's defaults; a mapping overrides them.
        self._step_cache_config = _StepCache.parse(model_config.get("step_cache"))
        self._step_cache_graphs = None
        self._zero_residual = None
        self._step_cache_stats = None
        # vllm-omni's TeaCache for this model (_TeaCache): skip whole steps whose timestep
        # embedding barely moved, reusing the last computed prediction. Off by default.
        self._tea_cache_config = _TeaCache.parse(model_config.get("tea_cache"))
        if self._tea_cache_config is not None and self._step_cache_config is not None:
            raise ValueError("tea_cache and step_cache are alternatives; enable at most one")
        self._host_time_embed = None
        self._tea_cache_stats = None
        self._tea_cache_plans: dict = {}

        if config.img_proj_type != "unet":
            raise ValueError(f"Unsupported img_proj_type: {config.img_proj_type}")

        # Bring-up / bisection knob: truncate the decoder stack. Weight loading, the
        # compiled graphs and the host loop all follow it, so a 2-layer run exercises the
        # whole path in minutes instead of a full cold compile. Output is meaningless.
        if "num_layers" in model_config:
            layers = int(model_config["num_layers"])
            logger.warning(
                "HunyuanImage3: truncating the decoder stack to %d of %d layers; "
                "generated images are NOT meaningful.",
                layers,
                config.num_hidden_layers,
            )
            config.num_hidden_layers = layers

        self.model = NeuronHunyuanImage3Transformer(
            config,
            use_nki_mlp=use_nki_mlp,
            use_nki_attention=use_nki_attention,
            moe_block_size=moe_block_size,
            moe_implementation=moe_implementation,
            use_denoise_attention_kernel=use_denoise_kernel,
            tp_sequence_parallel=bool(model_config.get("tp_sequence_parallel", False)),
        )
        # With CFG replicas (cfg_parallel_size: 2) a CFG request runs one branch per
        # replica. cfg_combine: in_graph (the default) gathers the two branches in the
        # step graph and runs the final layer over every rank; mixin leaves the combine
        # to NeuronCFGParallelMixin (cfg_mixin_pipeline.py), each replica's final layer
        # over its own TP group.
        self.cfg_parallel = self.model.cfg_replicas
        self._cfg_combine = str(model_config.get("cfg_combine", "in_graph"))
        if self._cfg_combine not in ("in_graph", "mixin"):
            raise ValueError(f"cfg_combine must be in_graph or mixin, got {self._cfg_combine!r}")

        hidden_size = int(config.hidden_size)
        latent_channels = int(config.vae["latent_channels"])
        self.time_embed = TimestepEmbedder(hidden_size=hidden_size)
        self.time_embed_2 = TimestepEmbedder(hidden_size=hidden_size)
        self.timestep_emb = TimestepEmbedder(hidden_size=hidden_size)
        self.patch_embed = UNetDown(
            patch_size=config.patch_size,
            emb_channels=hidden_size,
            in_channels=latent_channels,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=hidden_size,
        )
        self.final_layer = UNetUp(
            patch_size=config.patch_size,
            emb_channels=hidden_size,
            in_channels=hidden_size,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=latent_channels,
            out_norm=True,
        )

        # The token embedding and the VAE weights stay on the host: the embedding is a
        # single gather over a 133k x 4096 table (1.1 GB that would otherwise be
        # replicated on every core), and the VAE decodes once per request, on one rank.
        # With vae_device=neuron that rank copies the decoder onto its core on first use.
        self.wte = nn.Embedding(int(config.vocab_size), hidden_size)
        self.vae = self._build_vae(config)
        self.vae.to(self.vae_dtype)

        self._tkwrapper = TokenizerWrapper(model_path)
        self.image_processor = HunyuanImage3ImageProcessor(config)
        self.scheduler = self._build_scheduler()

        self.is_output_rank = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        self._compiled_denoise_step = None
        self._device_vae_decoder = None
        self._stage_seconds: dict[str, float] = {}
        self._last_perf_metrics: dict | None = None
        self._guidance_scale = None
        self._guidance_rescale = 0.0
        self._num_timesteps = None
        self._current_timestep = None
        self.skip_warmup = True
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=False,
        )

    # ---- construction helpers ----------------------------------------------

    @staticmethod
    def _build_vae(config):
        # vLLM-Omni's tile-parallel VAE for this model (DistributedVaeMixin), with the
        # tiles decoded on device; the engine enables it with
        # parallel_config.vae_patch_parallel_size > 1 (and vae_use_tiling).
        from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import (
            NeuronDistributedAutoencoderKLHunyuan,
        )

        return NeuronDistributedAutoencoderKLHunyuan.from_config(config.vae)

    def _build_scheduler(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=self.generation_config.flow_shift,
            use_dynamic_shifting=False,
            base_shift=0.5,
            max_shift=1.15,
            time_shift_type="exponential",
            stochastic_sampling=False,
        )

    @property
    def device(self) -> torch.device:
        """This rank's NeuronCore (overrides PreTrainedModel's parameter-derived property)."""
        return self._device

    @property
    def transformer(self):
        """Alias used by the runner's tensor-capture hooks and component discovery."""
        return self.model

    @property
    def pipeline(self):
        """Upstream's diffusers wrapper is unused here; the Neuron loop lives in forward()."""
        return self

    def to(self, *args, **kwargs):
        """Move only the device-resident modules; the embedding and the VAE stay on host."""
        self.model.to(*args, **kwargs)
        for name in _REPLICATED_PREFIXES:
            if self._unet_shard is not None and name in ("patch_embed", "final_layer"):
                # Only this rank's slices run; the whole modules (~0.5 GiB) stay on host.
                continue
            getattr(self, name).to(*args, **kwargs)
        if self._unet_shard is not None:
            self._unet_shard.to(*args, **kwargs)
        return self

    # ---- weights -----------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """A local directory for ``od_config.model``: as given, or its hub snapshot."""
        if getattr(self, "_model_path", None) is None:
            # The same resolver as the request pre-processor's (package __init__), so both
            # read od_config.revision.
            from vllm_omni_neuron.diffusion.models.hunyuan_image3 import resolve_snapshot

            self._model_path = resolve_snapshot(self.od_config.model, self.od_config.revision)
            # Resolved once per pipeline.
        return self._model_path

    def load_weights(self, weights=None) -> None:
        """Load the sharded backbone plus the replicated host/device modules.

        ``weights`` is the loader's generic iterator; it is ignored because each
        component needs its own reader (the Wan2.2 pipeline makes the same split).
        Returning ``None`` tells ``DiffusersPipelineLoader`` to skip its
        parameter-coverage check.
        """
        del weights
        model_path = self._resolve_model_path()
        self.model.load_weights(model_path)
        self._load_replicated_weights(model_path)
        if self._tea_cache_config is not None:
            import copy

            # The cache decision runs on the host; to() moves self.time_embed to the core.
            # Held outside the module tree so to() and the state dict never see it.
            self.__dict__["_host_time_embed"] = copy.deepcopy(self.time_embed).eval()
        model = self.model
        if self.unet_parallel and model.world_size > 1:
            # Over every rank of the stage, whatever the backbone's TP x CP split -- except
            # with the mixin's combine, where each CFG replica runs its own final layer.
            if self.cfg_parallel and self._cfg_combine == "mixin":
                rank, size, group = model.tp_rank, model.tp_size, model.tp_group
            else:
                rank, size, group = model.world_rank, model.world_size, model.world_group
            self._unet_shard = UNetShard(self.patch_embed, self.final_layer, rank, size)
            self._unet_shard.group = group
            logger.info("HunyuanImage3: patch embed / final layer split across %d ranks", size)

    def _replicated_weight_targets(self) -> dict[str, tuple[str, torch.Tensor, bool]]:
        """``{checkpoint key -> (parameter name, tensor, required)}`` for non-sharded modules.

        Buffers are optional: a module may carry derived state (e.g. a VAE's cached
        constants) that the checkpoint does not store.
        """
        targets: dict[str, tuple[str, torch.Tensor, bool]] = {}

        def add(module: nn.Module, prefix: str) -> None:
            for name, tensor in module.named_parameters():
                targets[f"{prefix}.{name}"] = (f"{prefix}.{name}", tensor, True)
            for name, tensor in module.named_buffers():
                targets.setdefault(f"{prefix}.{name}", (f"{prefix}.{name}", tensor, False))

        for prefix in _REPLICATED_PREFIXES:
            add(getattr(self, prefix), prefix)
        add(self.vae, "vae")
        targets["model.wte.weight"] = ("wte.weight", self.wte.weight, True)
        return targets

    def _load_replicated_weights(self, model_path: str) -> None:
        """Name-for-name load of the embedder / patch / final-layer / VAE weights."""
        from safetensors import safe_open

        targets = self._replicated_weight_targets()
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as handle:
                weight_map = json.load(handle)["weight_map"]
            files: dict[str, list[str]] = {}
            for key in targets:
                if key in weight_map:
                    files.setdefault(weight_map[key], []).append(key)
        else:
            files = {"model.safetensors": list(targets)}

        loaded: dict[str, torch.Tensor] = {}
        for file_name, keys in files.items():
            path = os.path.join(model_path, file_name)
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for key in keys:
                    if key not in available:
                        continue
                    param_name, tensor, _ = targets[key]
                    loaded[param_name] = handle.get_tensor(key).to(tensor.dtype)

        missing = sorted(
            {name for name, _, required in targets.values() if required} - set(loaded)
        )
        if missing:
            raise RuntimeError(
                f"HunyuanImage3: {len(missing)} replicated weights are missing from the "
                f"checkpoint; first few: {missing[:8]}"
            )

        self.load_state_dict(loaded, strict=False, assign=True)
        logger.info("HunyuanImage3: loaded %d replicated/host weights", len(loaded))

    # ---- compilation -------------------------------------------------------

    def _split_tokens(self, step_tokens: int, token_h: int, token_w: int) -> bool:
        """Whether a rank's ``step_tokens`` per step are its CP share of the step's tokens.

        With context parallelism a step either splits its tokens across the CP ranks or,
        with ``cfg_parallel``, gives each CP rank a whole CFG branch. The shapes the host
        passes (the RoPE table's length) say which, so each mode traces its own graph.
        """
        return self.model.cp_size > 1 and step_tokens < 1 + token_h * token_w

    def _sp_real_len(self, shard_len: int, token_h: int, token_w: int) -> int:
        """The step tokens behind a ``shard_len`` SP shard: a CP share, or the whole step."""
        model, total = self.model, 1 + token_h * token_w
        share = -(-total // model.cp_size)
        if model.cp_size > 1 and -(-share // model.tp_size) == shard_len:
            return share
        return total

    def _embed_step_inputs(self, latents: torch.Tensor, timestep: torch.Tensor, split=True):
        """Patch embed: ``[timestep token] + image tokens``, plus the token grid size."""
        t_emb = self.time_embed(timestep)
        shard = self._unet_shard
        if shard is not None:
            image_tokens, token_h, token_w = unet_down_tokens(shard, latents, t_emb, shard.all_gather)
        else:
            image_tokens, token_h, token_w = _unet_down_tokens(self.patch_embed, latents, t_emb)
        timestep_token = self.timestep_emb(timestep).reshape(latents.shape[0], 1, -1)
        tokens = torch.cat((timestep_token, image_tokens), dim=1)
        if self.model.cp_size > 1 and split:
            tokens = cp_local_tokens(tokens, self.model.cp_size, self.model.cp_rank)
        # tp_sequence_parallel: the backbone takes this TP rank's sequence shard.
        return self.model.sp_shard(tokens), token_h, token_w

    def _unembed_step_output(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor, token_h: int, token_w: int
    ) -> torch.Tensor:
        """Final layer: the image tokens back to a latent-shaped prediction."""
        shard = self._unet_shard
        t_emb = self.time_embed_2(timestep)
        model = self.model
        if model.sp_enabled:
            hidden_states = model.sp_unshard(
                hidden_states, self._sp_real_len(hidden_states.shape[1], token_h, token_w)
            )
        if self._split_tokens(hidden_states.shape[1], token_h, token_w):
            hidden_states = cp_gather_tokens(hidden_states, model.cp_size, model.cp_group)
        elif model.cp_size > 1 and self._cfg_combine == "in_graph":
            # CFG parallel: every replica's branch, as the CFG batch the final layer takes.
            hidden_states = cp_gather_batch(hidden_states, model.cp_size, model.cp_group)
        image_tokens = hidden_states[:, 1 : 1 + token_h * token_w, :]
        if shard is not None:
            return unet_up_image(shard, image_tokens, t_emb, token_h, token_w, shard.all_gather)
        return _unet_up_image(self.final_layer, image_tokens, t_emb, token_h, token_w)

    def _denoise_step_graph(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        timestep_key_bias: torch.Tensor,
        image_key_bias: torch.Tensor,
        *prompt_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Patch embed -> backbone denoise -> final layer, as one traced graph."""
        patch = int(self.hf_config.patch_size)
        split = self._split_tokens(
            cos.shape[1], latents.shape[2] // patch, latents.shape[3] // patch
        )
        hidden_states, token_h, token_w = self._embed_step_inputs(latents, timestep, split)
        hidden_states = self.model.forward_denoise(
            hidden_states,
            cos,
            sin,
            (timestep_key_bias, image_key_bias),
            *prompt_kv,
            split_tokens=split,
        )
        return self._unembed_step_output(hidden_states, timestep, token_h, token_w)

    # The three graphs of a step-cached denoise step (Cache-DiT's DBCache with one
    # compute block in front and none behind; see _StepCache). The host decides between
    # the last two from the scalar the first one returns.

    def _cache_front_graph(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        timestep_key_bias: torch.Tensor,
        image_key_bias: torch.Tensor,
        prompt_key: torch.Tensor,
        prompt_value: torch.Tensor,
        previous_residual: torch.Tensor,
    ):
        """Patch embed and the first layer; Cache-DiT's diff against the last full step.

        Returns the first layer's output, its residual, and
        ``mean|R_prev - R| / mean|R_prev|`` as a one-element tensor. Every TP rank holds
        the same (all-reduced) hidden states, so every rank computes the same diff; with
        context parallelism the CP ranks sum their live tokens' shares first.
        """
        patch = int(self.hf_config.patch_size)
        split = self._split_tokens(
            cos.shape[1], latents.shape[2] // patch, latents.shape[3] // patch
        )
        hidden_in, token_h, token_w = self._embed_step_inputs(latents, timestep, split)
        model = self.model
        hidden = model.denoise_layer(
            model.layers[0],
            hidden_in,
            cos,
            sin,
            (timestep_key_bias, image_key_bias),
            prompt_key,
            prompt_value,
            split,
        )
        residual = hidden - hidden_in
        previous = previous_residual.float()
        if model.sp_enabled:
            # Each rank holds a shard of its replica's (or CP rank's) tokens: count the
            # live ones and sum over every rank.
            local = hidden.shape[1]
            position = model.tp_rank * local + torch.arange(local, device=hidden.device)
            valid = position < cos.shape[1]
            if split:
                position = position + model.cp_rank * cos.shape[1]
            live = (valid & (position < 1 + token_h * token_w)).to(torch.float32).reshape(1, local, 1)
            delta, base = (previous - residual.float()).abs() * live, previous.abs() * live
            sums = torch.stack((delta.sum(), base.sum()))
            torch.distributed.all_reduce(sums, group=model.world_group)
            return hidden, residual, (sums[0] / sums[1]).reshape(1)
        if model.cp_size == 1:
            diff = (previous - residual.float()).abs().mean() / previous.abs().mean()
            return hidden, residual, diff.reshape(1)
        delta, base = (previous - residual.float()).abs(), previous.abs()
        if split:
            local = hidden.shape[1]
            position = model.cp_rank * local + torch.arange(local, device=hidden.device)
            live = (position < 1 + token_h * token_w).to(torch.float32).reshape(1, local, 1)
            delta, base = delta * live, base * live
        sums = torch.stack((delta.sum(), base.sum()))
        torch.distributed.all_reduce(sums, group=model.cp_group)
        return hidden, residual, (sums[0] / sums[1]).reshape(1)

    def _cache_rest_graph(
        self,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        timestep_key_bias: torch.Tensor,
        image_key_bias: torch.Tensor,
        token_h: int,
        token_w: int,
        *prompt_kv: torch.Tensor,
    ):
        """The remaining layers and the final layer; also their residual, for the cache."""
        first = hidden
        # The RoPE rows are this rank's tokens whatever the CP mode (hidden is an SP shard).
        split = self._split_tokens(cos.shape[1], token_h, token_w)
        for index, layer in enumerate(self.model.layers[1:]):
            hidden = self.model.denoise_layer(
                layer,
                hidden,
                cos,
                sin,
                (timestep_key_bias, image_key_bias),
                prompt_kv[2 * index],
                prompt_kv[2 * index + 1],
                split,
            )
        pred = self._unembed_step_output(hidden, timestep, token_h, token_w)
        return pred, hidden - first

    def _cache_hit_graph(
        self,
        hidden: torch.Tensor,
        block_residual: torch.Tensor,
        timestep: torch.Tensor,
        token_h: int,
        token_w: int,
    ) -> torch.Tensor:
        """A cached step: the first layer's output plus the remaining layers' last residual."""
        return self._unembed_step_output(hidden + block_residual, timestep, token_h, token_w)

    def compile(self, *args, **kwargs):
        """Compile the prefill graph and the fused denoise-step graph."""
        options = dict(kwargs.pop("options", {}) or {})
        options["compiler_args"] = [
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--hbm-scratchpad-page-size=2048",
        ]
        kwargs.setdefault("fullgraph", True)
        kwargs.setdefault("dynamic", False)
        # Kept for graphs built later, on first use (the AR decode step).
        self._compile_settings = (args, options, kwargs)
        self.model.compile(*args, options=options, **kwargs)
        self._compiled_denoise_step = torch.compile(
            self._denoise_step_graph,
            *args,
            options={**options, "model_name": "hunyuan_image3_denoise_step"},
            **kwargs,
        )
        if self._step_cache_config is not None:
            self._step_cache_graphs = tuple(
                torch.compile(
                    fn, *args, options={**options, "model_name": f"hunyuan_image3_{name}"}, **kwargs
                )
                for name, fn in (
                    ("cache_front", self._cache_front_graph),
                    ("cache_rest", self._cache_rest_graph),
                    ("cache_hit", self._cache_hit_graph),
                )
            )
        return self

    def _run_denoise_step(self, *call_args):
        fn = self._compiled_denoise_step or self._denoise_step_graph
        return fn(*call_args)

    def _run_cached_step(
        self,
        cache: "_StepCache",
        index: int,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        timestep_key_bias: torch.Tensor,
        image_key_bias: torch.Tensor,
        prompt_kv,
    ) -> torch.Tensor:
        """One denoise step through the step cache: front, then rest or hit."""
        front, rest, hit = self._step_cache_graphs or (
            self._cache_front_graph,
            self._cache_rest_graph,
            self._cache_hit_graph,
        )
        patch = int(self.hf_config.patch_size)
        token_h, token_w = latents.shape[2] // patch, latents.shape[3] // patch
        previous = cache.residual
        if previous is None:
            # This rank's tokens per step: one per RoPE row, whatever the CP mode (its
            # sequence shard of them with tp_sequence_parallel).
            rows = cos.shape[1]
            if self.model.sp_enabled:
                rows = -(-rows // self.model.tp_size)
            shape = (latents.shape[0], rows, int(self.hf_config.hidden_size))
            if self._zero_residual is None or tuple(self._zero_residual.shape) != shape:
                # Only read before the first full step, which never caches.
                self._zero_residual = torch.zeros(shape, dtype=self.model.dtype).to(self.device)
            previous = self._zero_residual
        hidden, residual, diff = front(
            latents,
            timestep,
            cos,
            sin,
            timestep_key_bias,
            image_key_bias,
            prompt_kv[0],
            prompt_kv[1],
            previous,
        )
        diff = float(diff.cpu()[0])
        use_cache = _all_tp_ranks(cache.wants_cache(index, diff))
        cache.record(index, diff, use_cache)
        if use_cache:
            return hit(hidden, cache.block_residual, timestep, token_h, token_w)
        pred, block_residual = rest(
            hidden,
            timestep,
            cos,
            sin,
            timestep_key_bias,
            image_key_bias,
            token_h,
            token_w,
            *prompt_kv[2:],
        )
        cache.update(residual, block_residual)
        return pred

    # ---- request preparation ----------------------------------------------

    def _key_biases(
        self, attention_mask: torch.Tensor, prompt_len: int, image_len: int, split_tokens=True
    ):
        """Derive the two additive key biases the denoise graph needs.

        ``attention_mask`` is upstream's ``[B, 1, S, S]`` boolean generation mask:
        lower-triangular, with a full-attention block over the generated-image span.
        The denoise step's queries and keys are the ``image_len`` positions starting at
        ``prompt_len`` — the timestep token followed by the ``<img>`` tokens. The
        template emits more tokens after that span (``<eoi>``, then the answer/bot
        suffixes), and those are *not* part of the step, exactly as upstream's
        steady-state step drops them; so every slice below is bounded by
        ``prompt_len + image_len`` rather than by the full sequence length.

        Row ``prompt_len`` is the timestep token and the rows after it are image tokens;
        both see the whole prompt, so each one's visibility over ``[prompt | image]``
        collapses to a single vector. Reading them out of the real mask — rather than
        assuming where the full-attention block starts — keeps this exact whichever side
        of the span the timestep token falls on.

        The biases cover the live keys only: a context-parallel split pads the step to
        ``cp`` equal slices, and attention drops the padding key before it reads the
        biases. Only CP rank 0 then holds the timestep token: the other
        ranks' first query is an image token, so their "timestep" bias is the image one.
        ``split_tokens=False`` (CFG parallel) builds the whole-step biases instead.
        """
        total = attention_mask.shape[-1]
        stop = prompt_len + image_len
        if image_len <= 1 or stop > total:
            raise ValueError(
                f"Unexpected generation layout: prompt_len={prompt_len}, "
                f"image_len={image_len}, total={total}"
            )
        mask = attention_mask[0, 0, prompt_len:stop, :stop].bool().cpu()

        # Only two mask rows are read; with CFG both batch rows must agree on them,
        # otherwise one branch would silently run with the other's visibility.
        for row in (0, 1):
            reference = mask[row]
            for branch in range(1, attention_mask.shape[0]):
                other = attention_mask[branch, 0, prompt_len + row, :stop].bool().cpu()
                if not bool(torch.all(other == reference)):
                    raise ValueError(
                        "The Neuron HunyuanImage3 path requires the CFG branches to share "
                        f"the generation mask (row {prompt_len + row} differs on branch "
                        f"{branch})."
                    )

        image_rows = mask[1:]
        if not bool(torch.all(image_rows == image_rows[:1])):
            raise ValueError(
                "The Neuron HunyuanImage3 path requires a uniform attention pattern "
                "across generated image tokens."
            )
        image_row = image_rows[0]
        timestep_row = mask[0]
        if self.model.layers[0].self_attn.use_denoise_kernel and not bool(
            torch.all(image_row[prompt_len:])
        ):
            raise ValueError(
                "denoise_attention: nki needs every generated image token to see the whole "
                "step (the timestep token and every image token); set denoise_attention: "
                "torch for this generation mask."
            )
        if not bool(torch.all(image_row[:prompt_len])) or not bool(
            torch.all(timestep_row[:prompt_len])
        ):
            raise ValueError(
                "The Neuron HunyuanImage3 path requires generated tokens to attend to "
                "the whole prompt prefix."
            )

        cp_size = self.model.cp_size if split_tokens else 1
        # The step's live keys only: a CP split's padding key is dropped before attention.
        columns = self.prefill_len + image_len

        def build(row: torch.Tensor) -> torch.Tensor:
            bias = torch.full((1, 1, 1, columns), _MASK_BIAS, dtype=torch.float32)
            bias[..., :prompt_len] = 0.0
            bias[0, 0, 0, self.prefill_len : self.prefill_len + image_len] = torch.where(
                row[prompt_len:],
                torch.zeros((), dtype=torch.float32),
                torch.full((), _MASK_BIAS, dtype=torch.float32),
            )
            return bias.to(self.device)

        if cp_size > 1 and self.model.cp_rank > 0:
            return build(image_row), build(image_row)
        return build(timestep_row), build(image_row)

    def _prompt_inputs(self, input_ids: torch.Tensor, cos, sin, prompt_len: int, cond_tokens=None):
        """Right-pad the prompt to ``prefill_len`` and build its embeddings and RoPE.

        Right-padding is safe because the prefill attends causally: the padded tail
        cannot influence any real prompt position. The pad columns are then masked out
        of every denoise step by the key biases.
        """
        bsz = input_ids.shape[0]
        pad_id = int(getattr(self.hf_config, "pad_token_id", 0) or 0)
        padded_ids = torch.full((bsz, self.prefill_len), pad_id, dtype=torch.long)
        padded_ids[:, :prompt_len] = input_ids[:, :prompt_len].cpu()
        embeds = self.wte(padded_ids).to(dtype=self.model.dtype)
        if cond_tokens is not None:
            # The conditioning images' VAE, timestep and ViT tokens over their placeholders.
            positions, values = cond_tokens
            embeds[:, positions] = values.to(embeds.dtype)
        if os.environ.get("HY3_DUMP_INPUTS"):
            self._dump = {"embeds": embeds.clone(), "cos": cos.clone(), "sin": sin.clone()}

        pad_cos = torch.zeros((bsz, self.prefill_len, cos.shape[-1]), dtype=torch.float32)
        pad_sin = torch.zeros((bsz, self.prefill_len, sin.shape[-1]), dtype=torch.float32)
        pad_cos[:, :prompt_len] = cos[:, :prompt_len].float().cpu()
        pad_sin[:, :prompt_len] = sin[:, :prompt_len].float().cpu()
        return (
            embeds.to(self.device),
            pad_cos.to(self.device),
            pad_sin.to(self.device),
        )

    def _denoise_rope(self, cos, sin, image_mask: torch.Tensor, prompt_len: int, split_tokens=True):
        """RoPE for ``[timestep token] + image tokens``, gathered by absolute position.

        With context parallelism and ``split_tokens``, only this rank's share of them.
        """
        bsz, seq_len = image_mask.shape
        index = torch.arange(seq_len, device=image_mask.device).unsqueeze(0).repeat(bsz, 1)
        image_positions = index.masked_select(image_mask.bool()).reshape(bsz, -1)
        timestep_positions = torch.full(
            (bsz, 1), prompt_len, dtype=image_positions.dtype, device=image_positions.device
        )
        positions = torch.cat((timestep_positions, image_positions), dim=1)
        step_cos = real_batched_index_select(cos, 1, positions).float()
        step_sin = real_batched_index_select(sin, 1, positions).float()
        if self.model.cp_size > 1 and split_tokens:
            step_cos, step_sin = (
                cp_local_tokens(table, self.model.cp_size, self.model.cp_rank)
                for table in (step_cos, step_sin)
            )
        return step_cos.to(self.device), step_sin.to(self.device)

    # ---- execution ---------------------------------------------------------

    def forward(
        self,
        req,
        prompt: str | list[str] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator=None,
        **kwargs,
    ) -> DiffusionOutput:
        """Generate one image: prefill the prompt once, then replay the denoise NEFF."""
        del kwargs
        # The engine's warmup request runs at its own resolution (512x512), which is a
        # different compiled shape than the one real requests use, so honouring it would
        # buy a second cold NEFF build and nothing else. Skip it the way the Wan2.2
        # pipeline does.
        if getattr(self, "skip_warmup", False) and getattr(req, "request_ids", None) == [
            "dummy_req_id"
        ]:
            first = req.prompts[0] if req.prompts else None
            prompt_text = first if isinstance(first, str) else (first or {}).get("prompt")
            if prompt_text == "dummy run":
                logger.info("Skipping warmup request on the Neuron HunyuanImage3 pipeline")
                return DiffusionOutput(output=None)

        start = time.perf_counter()
        sampling = req.sampling_params
        extra_args = getattr(sampling, "extra_args", {}) or {}
        (
            prompt_from_req,
            cot_text_list,
            system_prompt,
            batch_cond_image_info,
            tokenizer_bot_task,
        ) = self._extract_prompt_inputs(
            req.prompts, extra_args, request_id=req.request_id, allow_cond_image=True
        )
        prompt = prompt_from_req or prompt
        from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_ar import (
            AR_BOT_TASKS,
        )

        # A request's conditioning tokens are computed once and shared by its AR and DiT.
        self._cond_values_cache = None
        generator = sampling.generator or generator
        if generator is None:
            # Keep RNG on the host: torch.Generator(<neuron device>) is unsupported, and
            # a CPU generator reproduces the diffusers/GPU trajectory for a given seed.
            seeds = self.prepare_seed(seed=getattr(sampling, "seed", None), batch_size=1)
            generator = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]

        self._ar_stats = None
        bot_task = extra_args.get("bot_task")
        first = req.prompts[0] if req.prompts else None
        if bot_task is None and isinstance(first, dict):
            bot_task = first.get("bot_task")
        if bot_task is None:
            # vllm-omni 0.24's diffusion chat endpoint does not forward a request's
            # bot_task (only later releases pass model extras through extra_args), so a
            # served stage can set one for every request in model_config.
            bot_task = dict(self.od_config.model_config or {}).get("bot_task")
        if (
            bot_task in AR_BOT_TASKS
            and batch_cond_image_info
            and not any(text is not None for text in cot_text_list)
        ):
            # Image editing with reasoning, as the official generate_image runs it: the
            # unified system prompt unless the request names one, then think / recaption
            # over the prompt *and* the images (the "gen_text" template), then the image.
            if system_prompt is None:
                from vllm_omni.diffusion.models.hunyuan_image3.system_prompt import (
                    get_system_prompt,
                )

                system_prompt = (get_system_prompt("en_unified", bot_task, None) or "").strip()
            ar_inputs = self.prepare_model_inputs(
                prompt=prompt,
                system_prompt=system_prompt,
                mode="gen_text",
                bot_task=bot_task.split("_")[0],
                generator=generator,
                batch_cond_image_info=batch_cond_image_info,
                device=torch.device("cpu"),
            )
            cond_tokens = self._cond_image_tokens(ar_inputs, generator)
            self._ar_stats = self._ar_generator().generate_conditioned(
                ar_inputs, bot_task, cond_tokens
            )
            if self.model.cp_size > 1:
                self._ar_stats = _broadcast_from_rank0(self._ar_stats)
            if self.is_output_rank:
                logger.info(
                    "HunyuanImage3 AR (%s, %d image(s)): %d prompt + %d generated tokens, "
                    "%.2f s decode: %r",
                    bot_task,
                    len(batch_cond_image_info[0]),
                    self._ar_stats["prompt_tokens"],
                    self._ar_stats["generated_tokens"],
                    self._ar_stats["decode_seconds"],
                    self._ar_stats["text"][:2000],
                )
            cot_text_list = [self._ar_stats["text"]]
            tokenizer_bot_task = "auto"
        elif bot_task in AR_BOT_TASKS and not any(text is not None for text in cot_text_list):
            # Think / recaption on the same backbone first; the DiT then conditions on
            # the text, as upstream's ar2diffusion hands it over (bot_task back to auto).
            text_prompt = prompt[0] if isinstance(prompt, list) else prompt
            self._ar_stats = self._ar_generator().generate(text_prompt, bot_task, system_prompt)
            if self.model.cp_size > 1:
                # Each TP block decodes on its own; the DiT prompt must be the same on
                # every rank, so everyone takes rank 0's text.
                self._ar_stats = _broadcast_from_rank0(self._ar_stats)
            if self.is_output_rank:
                logger.info(
                    "HunyuanImage3 AR (%s): %d prompt + %d generated tokens, %.2f s decode, "
                    "%s s/token: %r",
                    bot_task,
                    self._ar_stats["prompt_tokens"],
                    self._ar_stats["generated_tokens"],
                    self._ar_stats["decode_seconds"],
                    self._ar_stats["seconds_per_token"],
                    self._ar_stats["text"][:2000],
                )
            cot_text_list = [self._ar_stats["text"]]
            tokenizer_bot_task = "auto"
        cot_text = (
            [self._normalize_cot_text(text) for text in cot_text_list]
            if any(text is not None for text in cot_text_list)
            else None
        )

        height = sampling.height or height
        width = sampling.width or width
        steps = sampling.num_inference_steps or num_inference_steps
        if steps is None:
            steps = int(self.generation_config.diff_infer_steps)
        if sampling.guidance_scale_provided:
            guidance = float(sampling.guidance_scale)
        elif guidance_scale is not None:
            guidance = float(guidance_scale)
        else:
            guidance = float(self.generation_config.diff_guidance_scale)


        model_inputs = self.prepare_model_inputs(
            prompt=prompt,
            cot_text=cot_text,
            system_prompt=system_prompt,
            mode="gen_image",
            generator=generator,
            image_size=(height, width),
            num_inference_steps=steps,
            guidance_scale=guidance,
            bot_task=tokenizer_bot_task,
            batch_cond_image_info=batch_cond_image_info,
            # Everything the templating path builds (ids, RoPE tables, masks) stays on
            # the host; this pipeline moves only the graph inputs to the device.
            device=torch.device("cpu"),
        )

        image = self._generate_image(model_inputs, steps=steps, guidance=guidance)
        elapsed = time.perf_counter() - start

        custom_output = {}
        if any(text is not None for text in cot_text_list):
            custom_output["ar_generated_text"] = cot_text_list[0]
        self._record_perf_metrics(steps=steps, height=height, width=width, e2e=elapsed)
        return DiffusionOutput(output=image, custom_output=custom_output)

    def _ar_generator(self):
        """The think / recaption generator, built on the first AR request. Every rank."""
        if getattr(self, "_ar", None) is None:
            from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_ar import (
                NeuronHunyuanImage3AR,
            )

            model_config = dict(self.od_config.model_config or {})
            self._ar = NeuronHunyuanImage3AR(
                self,
                self._resolve_model_path(),
                max_new_tokens=int(model_config.get("ar_max_new_tokens", 1024)),
                compile_kwargs=getattr(self, "_compile_settings", None),
            )
        return self._ar

    def _generate_image(self, model_inputs: dict[str, Any], *, steps: int, guidance: float):
        config = self.hf_config
        input_ids = model_inputs["input_ids"]
        image_mask = model_inputs["image_mask"]
        generator = model_inputs["generator"]
        batch_gen_image_info = model_inputs["batch_gen_image_info"]

        prompt_lens = model_inputs["gen_timestep_scatter_index"][:, -1]
        if not bool(torch.all(prompt_lens == prompt_lens[0])):
            raise ValueError(
                "The Neuron HunyuanImage3 path requires the generated timestep position "
                "to match across the CFG batch."
            )
        prompt_len = int(prompt_lens[0].item())
        # The smallest prompt bucket the prompt fits (prefill_buckets), for this request.
        self.prefill_len = self._prefill_bucket(prompt_len)

        attention_mask = self._prepare_attention_mask_for_generation(
            input_ids, self.generation_config, model_kwargs=model_inputs
        )
        # The denoise step covers the timestep token plus the <img> tokens, which is
        # what `image_mask` marks; the template's trailing <eoi>/suffix tokens are not
        # part of the step (upstream's steady-state step drops them too).
        image_len = 1 + int(image_mask[0].sum().item())
        cfg_factor = 1 + int(guidance > 1.0)
        # CFG parallel: this CP rank's TP block runs CFG branch cp_rank (0 = cond), whole.
        branch = self.model.cp_rank if self.cfg_parallel and cfg_factor == 2 else None
        timestep_key_bias, image_key_bias = self._key_biases(
            attention_mask, prompt_len, image_len, split_tokens=branch is None
        )

        cos, sin = self.get_pos_emb(model_inputs["custom_pos_emb"], model_inputs["position_ids"])

        prefill_start = time.perf_counter()
        def rows(tensor):
            # This rank's CFG branch of a batch-major tensor (shared rows stay as they are).
            if branch is None or tensor.shape[0] == 1:
                return tensor
            return tensor[branch : branch + 1]

        conditioned = model_inputs.get("cond_vae_image_mask") is not None
        cond_tokens = self._cond_image_tokens(model_inputs, generator) if conditioned else None
        prompt_embeds, prompt_cos, prompt_sin = self._prompt_inputs(
            rows(input_ids), rows(cos), rows(sin), prompt_len, cond_tokens
        )
        if conditioned:
            bound = self._prefill_bounds(attention_mask, prompt_len)
            if os.environ.get("HY3_DUMP_INPUTS") and self.is_output_rank:
                # Debugging aid: the prompt this rank prefills, for comparing against the
                # official implementation's inputs (token ids, embeddings, RoPE, bounds).
                torch.save(
                    {
                        **self._dump,
                        "input_ids": rows(input_ids).cpu(),
                        "prompt_len": prompt_len,
                        "bound": bound.cpu(),
                        "cond_positions": cond_tokens[0] if cond_tokens else None,
                        "cond_values": cond_tokens[1] if cond_tokens else None,
                    },
                    os.environ["HY3_DUMP_INPUTS"],
                )
            prompt_kv = self._run_prefill_bounded(prompt_embeds, prompt_cos, prompt_sin, bound)
        else:
            prompt_kv = self.model.run_prefill(prompt_embeds, prompt_cos, prompt_sin)
        prefill_seconds = time.perf_counter() - prefill_start

        step_cos, step_sin = self._denoise_rope(
            rows(cos), rows(sin), rows(image_mask), prompt_len, split_tokens=branch is None
        )

        image_info = batch_gen_image_info[0]
        downsample = config.vae_downsample_factor
        latents = torch.randn(
            (
                1,
                int(config.vae["latent_channels"]),
                int(image_info.image_height) // int(downsample[0]),
                int(image_info.image_width) // int(downsample[1]),
            ),
            generator=generator[0] if isinstance(generator, list) else generator,
            dtype=torch.float32,
            device="cpu",
        )

        timesteps, steps = retrieve_timesteps(self.scheduler, steps, "cpu", None, None)
        self._num_timesteps = len(timesteps)
        if cfg_factor != input_ids.shape[0]:
            raise ValueError(
                f"CFG factor {cfg_factor} does not match the templated batch "
                f"{input_ids.shape[0]}"
            )

        trace = _StepTrace.from_env(self.is_output_rank)
        if trace is not None:
            trace.record_inputs(
                model_inputs, attention_mask=attention_mask, guidance=guidance, steps=steps
            )
        step_cache = (
            _StepCache(**self._step_cache_config) if self._step_cache_config is not None else None
        )
        tea_cache = None
        if self._tea_cache_config is not None:
            tea_cache = _TeaCache(**self._tea_cache_config)
            # The plan depends only on the timesteps: compute it once per schedule.
            key = (tuple(float(t) for t in timesteps), tea_cache.threshold)
            if key not in self._tea_cache_plans:
                tea_cache.plan(self._host_time_embed, timesteps)
                self._tea_cache_plans[key] = (tea_cache.compute, tea_cache.distances)
            tea_cache.compute, tea_cache.distances = self._tea_cache_plans[key]
        tea_pred = None
        denoise_start = time.perf_counter()
        for index, timestep in enumerate(timesteps):
            if trace is not None:
                if trace.done(index):
                    break
                if trace.skip(index):
                    continue
                latents = trace.forced_latents(index, latents)
            self._current_timestep = timestep
            # Cast on the host and transfer separately: the Lite runtime's copy
            # requires matching dtypes, so a single .to(device=..., dtype=...) raises
            # "Expected self.dtype() == dst.dtype()".
            # Both CFG branches share the latents and timestep; with CFG parallel each rank
            # carries one row and the graph returns both branches' predictions.
            batch = 1 if branch is not None else cfg_factor
            model_input = latents.repeat(batch, 1, 1, 1).to(self.model.dtype)
            model_input = model_input.to(self.device)
            t_expand = timestep.repeat(batch).to(torch.float32).to(self.device)
            if tea_cache is not None and not tea_cache.compute[index]:
                # Skipped step: the last computed (CFG-batch) prediction stands in.
                pred = tea_pred
            elif step_cache is not None:
                pred = self._run_cached_step(
                    step_cache,
                    index,
                    model_input,
                    t_expand,
                    step_cos,
                    step_sin,
                    timestep_key_bias,
                    image_key_bias,
                    prompt_kv,
                )
            else:
                pred = self._run_denoise_step(
                    model_input,
                    t_expand,
                    step_cos,
                    step_sin,
                    timestep_key_bias,
                    image_key_bias,
                    *prompt_kv,
                )
            # Same rule in the other direction: copy first, then widen on the host.
            if pred is not tea_pred:
                pred = pred.cpu().to(torch.float32)
                tea_pred = pred
            raw_pred = pred
            if cfg_factor == 2:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + guidance * (pred_cond - pred_uncond)
            latents_in = latents
            latents = self.scheduler.step(pred, timestep, latents, return_dict=False)[0]
            if trace is not None:
                trace.record(index, timestep, latents_in, raw_pred, pred, latents)
            if index == 0:
                logger.info("HunyuanImage3: first denoise step complete (graph warm)")
        denoise_seconds = time.perf_counter() - denoise_start
        self._current_timestep = None
        self._step_cache_stats = step_cache.stats() if step_cache is not None else None
        self._tea_cache_stats = tea_cache.stats() if tea_cache is not None else None
        if tea_cache is not None and self.is_output_rank:
            logger.info("HunyuanImage3 TeaCache: %s", self._tea_cache_stats)
        if step_cache is not None and self.is_output_rank:
            logger.info("HunyuanImage3 step cache: %s", self._step_cache_stats)

        decode_start = time.perf_counter()
        # Only the rank whose output the engine keeps needs pixels. Decoding on all 32
        # ranks is pure duplicated work, and because the Lite worker pins
        # torch.set_num_threads(1) they also contend for the host instead of sharing it.
        vae_parts, vae_rank = self._vae_split()
        if self._vae_patch_parallel():
            # Tile-parallel decode: every rank takes part in the executor's collectives.
            image = self._decode(latents, generator)
        elif vae_parts > 1:
            # The decode is split by rows over ranks 0..parts-1, which all take part in
            # it; rank 0 (the output rank) gets the image.
            image = self._decode(latents, generator) if vae_rank is not None else None
        else:
            image = self._decode(latents, generator) if self.is_output_rank else None
        self._stage_seconds = {
            "prefill_seconds": prefill_seconds,
            "denoise_seconds": denoise_seconds,
            "vae_decode_seconds": time.perf_counter() - decode_start,
        }
        return image

    def _decode(self, latents: torch.Tensor, generator):
        """Decode latents to a PIL image, on the host or this rank's core. Output rank only."""
        from diffusers.image_processor import VaeImageProcessor

        # The Lite worker pins torch to one thread so N workers do not oversubscribe the
        # host. Only this rank decodes, so give the convolutions the box for its duration.
        previous_threads = torch.get_num_threads()
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (
            os.cpu_count() or 1
        )
        torch.set_num_threads(max(1, cpus))
        try:
            return self._decode_impl(latents, generator, VaeImageProcessor)
        finally:
            torch.set_num_threads(previous_threads)

    def _device_decoder(self):
        """The staged on-device decoder, built on first use. Output rank only.

        Built lazily rather than at load time because only the output rank decodes, and
        its graphs compile on the first call for each latent shape anyway. With
        ``vae_device_core`` set it runs in a worker process on that core
        (:mod:`.vae_worker`); otherwise on this rank's own core, next to its backbone
        shard -- which float16 does not fit beside.
        """
        if self._device_vae_decoder is None:
            # float16 is the default: same speed as bfloat16, 66 dB PSNR to the host
            # decode against 59. Read here, below the traced functions, so the options
            # do not move their lines (and with them the denoise graph's cache key).
            model_config = dict(self.od_config.model_config or {})
            dtype = str(model_config.get("vae_device_dtype", "float16"))
            if dtype not in ("float16", "bfloat16"):
                raise ValueError(f"vae_device_dtype must be float16 or bfloat16, got {dtype!r}")
            core = model_config.get("vae_device_core")
            parts, rank = self._vae_split()
            if parts > 1:
                from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (
                    ShardedDeviceDecoder,
                )

                self._device_vae_decoder = ShardedDeviceDecoder(
                    self.vae,
                    self.hf_config.vae,
                    self.device,
                    getattr(torch, dtype),
                    rank,
                    parts,
                    self._vae_group,
                )
            elif core is not None:
                from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_worker import (
                    RemoteDeviceDecoder,
                )

                self._device_vae_decoder = RemoteDeviceDecoder(
                    self._resolve_model_path(), self.hf_config.vae, int(core), dtype
                )
            else:
                from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (
                    StagedDeviceDecoder,
                )

                self._device_vae_decoder = StagedDeviceDecoder(
                    self.vae, self.hf_config.vae, self.device, dtype=getattr(torch, dtype)
                )
        return self._device_vae_decoder

    def _decode_impl(self, latents: torch.Tensor, generator, VaeImageProcessor):
        vae_config = self.vae.config
        on_device = self.vae_device == "neuron"
        # The device decoder casts to its own dtype; scale in float32 before it does.
        latents = latents.to(dtype=torch.float32 if on_device else self.vae_dtype)
        if getattr(vae_config, "scaling_factor", None):
            latents = latents / vae_config.scaling_factor
        if getattr(vae_config, "shift_factor", None):
            latents = latents + vae_config.shift_factor
        temporal = hasattr(self.vae, "ffactor_temporal")
        if temporal:
            latents = latents.unsqueeze(2)
        if on_device and self._vae_patch_parallel():
            return self._decode_tiles(latents, VaeImageProcessor)
        decoder = None
        if on_device:
            try:
                decoder = self._device_decoder()
            except Exception:
                logger.exception("HunyuanImage3: device VAE unavailable; using the host")
        if decoder is not None and not decoder.supports(latents):
            logger.warning(
                "HunyuanImage3: latents %s exceed the device VAE's spatial range; "
                "decoding this request on the host",
                tuple(latents.shape),
            )
            decoder = None
        if self._vae_split()[0] > 1:
            # The split decode's stages run collectives over its ranks, so they all take
            # the device path or all fall back to rank 0's host decode: a rank whose
            # decoder failed to build would otherwise leave the others waiting.
            ready = torch.tensor([int(decoder is not None)], dtype=torch.int32)
            torch.distributed.all_reduce(
                ready, op=torch.distributed.ReduceOp.MIN, group=self._vae_host_group
            )
            if not ready.item():
                decoder = None
        image = None
        with torch.no_grad():
            if decoder is not None:
                try:
                    image = decoder.decode(latents)
                except Exception:
                    # A failed device decode (a dead worker, device OOM) costs this request
                    # the host decode's 29 s, not the image; the next request rebuilds.
                    logger.exception("HunyuanImage3: device VAE decode failed; using the host")
                    self._device_vae_decoder = None
            if image is None and not self.is_output_rank:
                return None  # a split decode's other rank: the host fallback is rank 0's
            if image is None:
                host_latents = latents.to(dtype=self.vae_dtype)
                image = self.vae.decode(host_latents, return_dict=False, generator=generator)[0]
        if temporal:
            image = image.squeeze(2)
        if not self.is_output_rank:
            # A rank of a split decode: its part is done; only the output rank keeps pixels.
            return None

        processor = VaeImageProcessor(
            vae_scale_factor=int(self.hf_config.vae_downsample_factor[0])
        )
        images = processor.postprocess(
            image.to(torch.float32), output_type="pil", do_denormalize=[True] * image.shape[0]
        )
        return images[0]

    # ---- conditioning images (image editing / IT2I) -----------------------

    def _prefill_bucket(self, prompt_len: int) -> int:
        """The smallest prompt bucket that fits: ``prefill_len`` or one of ``prefill_buckets``.

        Each bucket is its own prefill and denoise graphs, so a short text prompt keeps
        its 512 bucket while a prompt carrying a conditioning image (~4096 VAE + up to
        1024 ViT tokens at 1024x1024) takes a larger one.
        """
        for bucket in self._prefill_buckets:
            if prompt_len <= bucket:
                return bucket
        raise ValueError(
            f"Prompt needs {prompt_len} tokens but the largest prompt bucket is "
            f"{self._prefill_buckets[-1]}. "
            "Add a larger one to engine_args.model_config.prefill_buckets (one more prefill "
            "and denoise compile)."
        )

    def _prefill_bounds(self, attention_mask: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """``[prefill_len]`` float32: prompt query ``i`` sees prompt keys ``[0, bound[i])``.

        Upstream's generation mask is causal with full attention inside each image block
        (``joint_full`` for conditioning images), so every prompt row is a prefix of the
        prompt -- checked here, along with the CFG branches agreeing. Padding rows stay
        causal; their keys are masked out of every later step by the key biases.
        """
        prompt = attention_mask[:, 0, :prompt_len, :prompt_len].bool().cpu()
        if not bool(torch.all(prompt == prompt[:1])):
            raise ValueError("The CFG branches must share the prompt's attention pattern.")
        rows = prompt[0]
        bound = rows.sum(dim=-1)
        prefix = torch.arange(prompt_len).reshape(1, -1) < bound.reshape(-1, 1)
        if not bool(torch.all(rows == prefix)) or bool(
            attention_mask[0, 0, :prompt_len, prompt_len:].any()
        ):
            raise ValueError(
                "The Neuron HunyuanImage3 prefill needs every prompt token to attend to a "
                "prefix of the prompt (causal text, full-attention image blocks)."
            )
        full = torch.arange(1, self.prefill_len + 1, dtype=torch.float32)
        full[:prompt_len] = bound.to(torch.float32)
        widened = (bound > torch.arange(1, prompt_len + 1)).nonzero().flatten()
        if widened.numel():
            logger.info(
                "HunyuanImage3 prefill bounds: %d of %d prompt rows see past themselves, rows "
                "%d..%d, bounds %s",
                widened.numel(),
                prompt_len,
                int(widened[0]),
                int(widened[-1]),
                sorted(set(bound[widened].tolist()))[:8],
            )
        return full.to(self.device)

    def _prefill_bounded_graph(self, inputs_embeds, cos, sin, bound):
        """The prompt prefill with per-query key bounds (:func:`forward_prefill_bounded`)."""
        from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
            forward_prefill_bounded,
        )

        return forward_prefill_bounded(self.model, inputs_embeds, cos, sin, bound)

    def _run_prefill_bounded(self, inputs_embeds, cos, sin, bound):
        if getattr(self, "_compiled_prefill_bounded", None) is None:
            settings = getattr(self, "_compile_settings", None)
            fn = self._prefill_bounded_graph
            if settings is not None:
                args, options, kwargs = settings
                fn = torch.compile(
                    fn,
                    *args,
                    options={**options, "model_name": "hunyuan_image3_prefill_bounded"},
                    **kwargs,
                )
            self._compiled_prefill_bounded = fn
        return self._compiled_prefill_bounded(inputs_embeds, cos, sin, bound)

    def _encode_cond_image(self, batch_cond_image_info_list, cfg_factor: int = 1, generator=None):
        """Defer upstream's per-rank encode: return the preprocessed pixels, unencoded.

        ``prepare_model_inputs`` runs on every rank; :meth:`_cond_image_tokens` encodes
        the images once, on rank 0. The CFG branches share them, so only row 0 is kept.
        """
        del cfg_factor, generator
        infos = batch_cond_image_info_list[0]
        pixels = [info.vae_image_info.image_tensor for info in infos]
        vit = torch.cat([info.vision_image_info.image_tensor for info in infos], dim=0)
        return pixels, None, [vit]

    def _cond_image_tokens(self, model_inputs: dict[str, Any], generator):
        """``(positions, values)`` of the conditioning images' tokens in the prompt.

        In prompt order: each image's VAE tokens (patch embed of its latents at t=0),
        its timestep token, then each image's SigLIP2 tokens (through the aligner), at
        the placeholders upstream's template marks. Rank 0 computes them on the host and
        every rank receives them over gloo: they are the same for both CFG branches, and
        32 single-threaded ranks recomputing them would take minutes.
        """
        masks = {
            key: model_inputs[key].cpu()
            for key in ("cond_vae_image_mask", "cond_vit_image_mask", "cond_timestep_scatter_index")
        }
        for key, mask in masks.items():
            if not bool(torch.all(mask == mask[:1])):
                raise ValueError(f"The CFG branches must share the conditioning layout ({key}).")
        vae_positions = masks["cond_vae_image_mask"][0].nonzero().flatten()
        timestep_positions = masks["cond_timestep_scatter_index"][0].flatten().long()
        vit_positions = masks["cond_vit_image_mask"][0].nonzero().flatten()
        hidden = int(self.hf_config.hidden_size)
        counts = (vae_positions.numel(), timestep_positions.numel(), vit_positions.numel())
        positions = torch.cat((vae_positions, timestep_positions, vit_positions))
        key = (tuple(id(pixels) for pixels in model_inputs["cond_vae_images"]), counts)
        cached = getattr(self, "_cond_values_cache", None)
        if cached is not None and cached[0] == key:
            return positions, cached[1]
        values = torch.empty((sum(counts), hidden), dtype=torch.float32)

        distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        me = torch.distributed.get_rank() if distributed else 0
        encode_rank = self._vae_encode_rank()
        source = encode_rank if encode_rank is not None else 0
        moments = None
        if me == encode_rank:
            # This rank encodes on its own core, then builds the tokens for everyone.
            start = time.perf_counter()
            moments = [self._device_encoder().encode(p) for p in model_inputs["cond_vae_images"]]
            encode_seconds = time.perf_counter() - start
        if me == source:
            start = time.perf_counter()
            values = self._compute_cond_tokens(model_inputs, generator, counts, moments)
            if moments is not None:
                logger.info("HunyuanImage3: device VAE encode in %.2fs", encode_seconds)
            logger.info(
                "HunyuanImage3: %d conditioning image(s), %d VAE + %d timestep + %d ViT tokens "
                "on the host in %.1fs",
                len(model_inputs["cond_vae_images"]),
                *counts,
                time.perf_counter() - start,
            )
        if distributed:
            torch.distributed.broadcast(values, src=source, group=get_world_group().cpu_group)
        self._cond_values_cache = (key, values)
        return positions, values

    def _compute_cond_tokens(self, model_inputs, generator, counts, moments=None) -> torch.Tensor:
        """The conditioning tokens, float32, on the host (the source rank only).

        ``moments`` are the device encoder's posterior moments per image; without them
        the host VAE encodes.
        """
        clock = time.perf_counter()
        modules = self._host_cond_modules()
        timings = {"modules": time.perf_counter() - clock}
        hidden = int(self.hf_config.hidden_size)
        previous_threads = torch.get_num_threads()
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        torch.set_num_threads(max(1, cpus))
        # The VAE sample draws from a copy of the request's generator, so every rank's
        # generator -- and with it the initial latents -- stays in the same state.
        source = generator[0] if isinstance(generator, list) else generator
        sample_generator = torch.Generator("cpu")
        sample_generator.set_state(source.get_state())
        try:
            with torch.no_grad():
                config = self.vae.config
                vae_tokens = []
                for index, pixels in enumerate(model_inputs["cond_vae_images"]):
                    clock = time.perf_counter()
                    if moments is not None:
                        from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import (
                            DiagonalGaussianDistribution,
                        )

                        encoded = DiagonalGaussianDistribution(moments[index])
                        if os.environ.get("HY3_CHECK_DEVICE_ENCODE"):
                            image = pixels.cpu().to(self.vae_dtype).reshape(1, *pixels.shape[-3:])
                            host = self.vae.encode(image.unsqueeze(2)).latent_dist.parameters.float()
                            error = (moments[index] - host).norm() / host.norm()
                            logger.info(
                                "HunyuanImage3: device VAE encode vs host float32: moments "
                                "relative %.3e (mean part %.3e)",
                                error.item(),
                                ((moments[index][:, : host.shape[1] // 2] - host[:, : host.shape[1] // 2]).norm()
                                 / host[:, : host.shape[1] // 2].norm()).item(),
                            )
                    else:
                        image = pixels.cpu().to(self.vae_dtype).reshape(1, *pixels.shape[-3:])
                        encoded = self.vae.encode(image.unsqueeze(2)).latent_dist
                    timings["vae_encode"] = timings.get("vae_encode", 0.0) + time.perf_counter() - clock
                    latents = encoded.sample(sample_generator).float()
                    if getattr(config, "shift_factor", None):
                        latents = latents - config.shift_factor
                    if getattr(config, "scaling_factor", None):
                        latents = latents * config.scaling_factor
                    latents = latents.squeeze(2)
                    clock = time.perf_counter()
                    t_emb = modules["time_embed"](torch.zeros(1))
                    tokens, _, _ = _unet_down_tokens(modules["patch_embed"], latents, t_emb)
                    timings["patch_embed"] = timings.get("patch_embed", 0.0) + time.perf_counter() - clock
                    vae_tokens.append(tokens.reshape(-1, hidden))
                timestep_tokens = modules["timestep_emb"](torch.zeros(counts[1])).reshape(-1, hidden)

                clock = time.perf_counter()
                vit_images = model_inputs["cond_vit_images"][0].cpu().float()
                vit_kwargs = model_inputs["vit_kwargs"]
                attention_mask = vit_kwargs["attention_mask"][0].cpu()
                spatial_shapes = vit_kwargs["spatial_shapes"][0].cpu()
                features = modules["vision"](
                    pixel_values=vit_images,
                    pixel_attention_mask=attention_mask,
                    spatial_shapes=spatial_shapes,
                ).last_hidden_state
                # Upstream's packed encoder leaves padding patches at zero.
                features = features * attention_mask.unsqueeze(-1).to(features.dtype)
                vit_tokens = modules["aligner"](features).reshape(-1, hidden)
                timings["vit"] = time.perf_counter() - clock
            values = torch.cat((*vae_tokens, timestep_tokens, vit_tokens)).float()
            logger.debug(
                "HunyuanImage3 conditioning token stats: VAE mean %.4f std %.4f | timestep std %.4f | "
                "ViT std %.4f | positions VAE %s.., timestep %s, ViT %s..",
                torch.cat(vae_tokens).mean().item(),
                torch.cat(vae_tokens).std().item(),
                timestep_tokens.std().item(),
                vit_tokens.std().item(),
                model_inputs["cond_vae_image_mask"][0].nonzero().flatten()[:3].tolist(),
                model_inputs["cond_timestep_scatter_index"][0].tolist(),
                model_inputs["cond_vit_image_mask"][0].nonzero().flatten()[:3].tolist(),
            )
        finally:
            torch.set_num_threads(previous_threads)
        logger.info(
            "HunyuanImage3 conditioning tokens on the host (%d threads): %s",
            max(1, cpus),
            {k: round(v, 2) for k, v in timings.items()},
        )
        if values.shape[0] != sum(counts):
            raise ValueError(
                f"Conditioning tokens {values.shape[0]} do not match the template's {sum(counts)} "
                f"placeholders (VAE/timestep/ViT {counts})."
            )
        return values

    def _vae_encode_rank(self) -> int | None:
        """The rank that encodes conditioning images on device, or ``None`` for the host.

        ``model_config.vae_encode_device: neuron`` (the default with a device VAE) runs
        the whole encoder on one rank's core, beside its backbone shard: rank
        ``vae_encode_rank``, by default the first rank the split decode leaves alone
        (rank 2 with ``vae_parallel: 2``). One core, no collectives: a device group
        without global rank 0 cannot bootstrap while rank 0 is idle, which is what
        ruled out splitting the encode over ranks 2 and 3 the way the decode is split.
        """
        parts, _ = self._vae_split()
        model_config = dict(self.od_config.model_config or {})
        device = str(model_config.get("vae_encode_device", "neuron"))
        if device not in ("neuron", "cpu"):
            raise ValueError(f"vae_encode_device must be 'neuron' or 'cpu', got {device!r}")
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = int(model_config.get("vae_encode_rank", parts if parts > 1 else 0))
        if device == "cpu" or self.vae_device != "neuron" or rank >= world:
            return None
        return rank

    def _device_encoder(self):
        """The device VAE encoder on this rank's core, built on its first image request."""
        if getattr(self, "_device_vae_encoder", None) is None:
            from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (
                StagedDeviceEncoder,
            )

            model_config = dict(self.od_config.model_config or {})
            dtype = getattr(torch, str(model_config.get("vae_device_dtype", "float16")))
            self._device_vae_encoder = StagedDeviceEncoder(
                self.vae, self.hf_config.vae, self.device, dtype
            )
        return self._device_vae_encoder

    def _host_cond_modules(self) -> dict:
        """Float32 host copies for the conditioning path, built on the first image request.

        SigLIP2 is transformers' ``Siglip2VisionModel`` (the checkpoint's
        ``vision_model.*`` uses its names) and the aligner the checkpoint's
        Linear-GELU-Linear; the timestep embedders are copied back from the device and
        the patch embed is the host module the UNet split was sliced from.
        """
        if getattr(self, "_cond_modules", None) is None:
            import copy

            from safetensors import safe_open
            from transformers import Siglip2VisionConfig, Siglip2VisionModel

            config = self.hf_config
            vit_config = dict(config.vit) if not hasattr(config.vit, "to_dict") else config.vit.to_dict()
            architecture = (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_channels",
                "num_patches",
                "patch_size",
                "hidden_act",
                "layer_norm_eps",
                "attention_dropout",
            )
            vision = Siglip2VisionModel(
                Siglip2VisionConfig(**{k: vit_config[k] for k in architecture if k in vit_config})
            )
            aligner_config = config.vit_aligner if isinstance(config.vit_aligner, dict) else vars(config.vit_aligner)
            if aligner_config.get("projector_type") != "mlp_gelu" or int(aligner_config.get("depth", 2)) != 2:
                raise ValueError(f"Unsupported vit_aligner {aligner_config}")
            width, hidden = int(aligner_config["input_dim"]), int(aligner_config["n_embed"])
            aligner = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden))

            model_path = self._resolve_model_path()
            with open(os.path.join(model_path, "model.safetensors.index.json")) as handle:
                weight_map = json.load(handle)["weight_map"]
            targets = {}
            for prefix, module in (("vision_model.", vision), ("vision_aligner.layers.", aligner)):
                for name in module.state_dict():
                    targets[prefix + name] = (module, name)
            state = {id(vision): {}, id(aligner): {}}
            files: dict[str, list[str]] = {}
            for key in targets:
                if key not in weight_map:
                    raise KeyError(f"conditioning weight {key} missing from the checkpoint")
                files.setdefault(weight_map[key], []).append(key)
            for filename, keys in files.items():
                with safe_open(os.path.join(model_path, filename), framework="pt", device="cpu") as f:
                    for key in keys:
                        module, name = targets[key]
                        state[id(module)][name] = f.get_tensor(key).float()
            vision.load_state_dict(state[id(vision)], strict=True)
            aligner.load_state_dict(state[id(aligner)], strict=True)

            def host_copy(module):
                host = TimestepEmbedder(hidden_size=int(config.hidden_size))
                host.load_state_dict({k: v.cpu().float() for k, v in module.state_dict().items()})
                return host.eval()

            self.__dict__["_cond_modules"] = {
                "vision": vision.eval(),
                "aligner": aligner.eval(),
                "time_embed": host_copy(self.time_embed),
                "timestep_emb": host_copy(self.timestep_emb),
                "patch_embed": copy.deepcopy(self.patch_embed).float().eval(),
            }
        return self._cond_modules

    def _vae_patch_parallel(self) -> bool:
        """Whether the decode is vLLM-Omni's tile-parallel one (vae_patch_parallel_size > 1)."""
        enabled = self.vae_device == "neuron" and self.vae.is_distributed_enabled()
        if enabled and int(dict(self.od_config.model_config or {}).get("vae_parallel", 1)) > 1:
            raise ValueError(
                "vae_parallel (the row split) and parallel_config.vae_patch_parallel_size "
                "(vLLM-Omni's tile split) are alternatives; set one"
            )
        return enabled

    def _decode_tiles(self, latents: torch.Tensor, VaeImageProcessor):
        """The tile-parallel device decode of scaled ``(B, C, T, h, w)`` latents. Every rank."""
        from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import StagedDeviceDecoder

        dtype = getattr(torch, str(dict(self.od_config.model_config or {}).get("vae_device_dtype", "float16")))
        self.vae.set_tile_decoder_factory(
            lambda: StagedDeviceDecoder(self.vae, self.hf_config.vae, self.device, dtype=dtype)
        )
        host = latents.to(dtype)
        z = host.to(self.device)
        with torch.no_grad():
            image = self.vae.spatial_tiled_decode(z, host_latent=host)
        if not self.is_output_rank:
            return None
        image = image.to("cpu").float().squeeze(2)
        processor = VaeImageProcessor(vae_scale_factor=int(self.hf_config.vae_downsample_factor[0]))
        return processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])[0]

    def _vae_split(self) -> tuple[int, int | None]:
        """``(parts, this rank's part or None)`` for ``model_config.vae_parallel``.

        ``vae_parallel: N`` splits the device decode by image rows over ranks 0..N-1 --
        neighbouring cores of the first chip -- instead of running it whole on one core
        (``vae_device_core``), which at TP16 x CP2 no longer fits beside a backbone shard.
        The groups are built on first use, by every rank (``new_group`` is collective).
        """
        if getattr(self, "_vae_split_cache", None) is None:
            model_config = dict(self.od_config.model_config or {})
            parts = int(model_config.get("vae_parallel", 1))
            if parts > 1 and self.vae_device != "neuron":
                # A split of the device decode; the host decode runs whole on rank 0.
                logger.info("HunyuanImage3: vae_device=%s, so vae_parallel is unused", self.vae_device)
                parts = 1
            if parts > 1 and model_config.get("vae_device_core") is not None:
                raise ValueError("vae_parallel and vae_device_core are alternatives; set one")
            rank = None
            self._vae_group = self._vae_host_group = None
            if parts > 1:
                from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (
                    vae_row_groups,
                )

                self._vae_group, self._vae_host_group = vae_row_groups(parts)
                me = torch.distributed.get_rank()
                rank = me if me < parts else None
            self._vae_split_cache = (parts, rank)
        return self._vae_split_cache

    # ---- metrics -----------------------------------------------------------

    def _record_perf_metrics(self, *, steps: int, height: int, width: int, e2e: float) -> None:
        stages = self._stage_seconds or {}
        denoise = stages.get("denoise_seconds") or 0.0
        metrics = {
            "prefill_seconds": stages.get("prefill_seconds"),
            "denoise_seconds": denoise,
            "vae_decode_seconds": stages.get("vae_decode_seconds"),
            "e2e_forward_seconds": e2e,
            "seconds_per_step": denoise / steps if steps else 0.0,
            "num_steps": steps,
            "height": height,
            "width": width,
            "prefill_len": self.prefill_len,
            "vae_device": self.vae_device,
            "lite_runtime": is_lite_runtime(),
        }
        ar = getattr(self, "_ar_stats", None)
        if ar is not None:
            metrics.update(
                ar_prompt_tokens=ar["prompt_tokens"],
                ar_generated_tokens=ar["generated_tokens"],
                ar_prefill_seconds=ar["prefill_seconds"],
                ar_decode_seconds=ar["decode_seconds"],
                ar_seconds_per_token=ar["seconds_per_token"],
                ar_stopped=ar["stopped"],
            )
        if self._step_cache_stats is not None:
            metrics["step_cache"] = self._step_cache_stats
        if self._tea_cache_stats is not None:
            metrics["tea_cache"] = self._tea_cache_stats
        self._last_perf_metrics = metrics
        if not self.is_output_rank:
            return
        metrics_dir = os.path.join(os.environ.get("WORKLOAD_OUTPUT_RW", "/tmp"), "metrics")
        try:
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir, "pipeline_perf_metrics.json"), "w") as handle:
                json.dump(metrics, handle, indent=2)
        except OSError as error:
            logger.warning("Failed to write perf metrics: %s", error)
        logger.info("HunyuanImage3 perf metrics: %s", metrics)


class _StepTrace:
    """Debug-only per-step capture of the denoise loop, for comparing implementations.

    Off unless ``HY3_TRACE_DIR`` or ``HY3_TRACE_FORCE_DIR`` is set. ``HY3_TRACE_DIR``: the
    output rank saves the request's inputs and each step's input latents, the raw
    CFG-batch prediction, the combined one and the scheduler's updated latents.
    ``HY3_TRACE_FORCE_DIR``: every step starts from the latents a previous trace recorded
    there instead of its own -- teacher forcing, so two implementations are compared on
    identical inputs at every step rather than on diverging trajectories.
    ``HY3_TRACE_STEPS`` (``0,25,49``): record only those steps; with forcing, compute
    only those.
    """

    def __init__(self, out_dir, force_dir, steps, save):
        self.out_dir, self.force_dir, self.steps, self.save = out_dir, force_dir, steps, save
        if out_dir and save:
            os.makedirs(out_dir, exist_ok=True)

    @classmethod
    def from_env(cls, is_output_rank):
        out_dir = os.environ.get("HY3_TRACE_DIR")
        force_dir = os.environ.get("HY3_TRACE_FORCE_DIR")
        if not out_dir and not force_dir:
            return None
        raw = os.environ.get("HY3_TRACE_STEPS")
        steps = sorted({int(x) for x in raw.split(",")}) if raw else None
        logger.warning(
            "HunyuanImage3 step trace: out=%s force=%s steps=%s", out_dir, force_dir, steps
        )
        return cls(out_dir, force_dir, steps, is_output_rank)

    def done(self, index):
        return bool(self.force_dir) and self.steps is not None and index > self.steps[-1]

    def skip(self, index):
        return bool(self.force_dir) and self.steps is not None and index not in self.steps

    def forced_latents(self, index, latents):
        if not self.force_dir:
            return latents
        return torch.load(os.path.join(self.force_dir, f"step_{index:02d}.pt"))["latents_in"]

    def record_inputs(self, model_inputs, **settings):
        """``inputs.pt``: what a reference needs to replay the request's steps.

        The templated inputs the official implementation's forward takes, plus
        ``settings`` (the attention mask, guidance, steps).
        """
        if self.out_dir and self.save:
            keys = ("input_ids", "image_mask", "position_ids", "gen_timestep_scatter_index", "cond_vae_images")
            kept = {key: model_inputs.get(key) for key in keys}
            torch.save({**kept, **settings}, os.path.join(self.out_dir, "inputs.pt"))

    def record(self, index, timestep, latents, raw_pred, pred, latents_out):
        if self.out_dir and self.save and (self.steps is None or index in self.steps):
            torch.save(
                {
                    "timestep": timestep,
                    "latents_in": latents,
                    "pred_raw": raw_pred,
                    "pred": pred,
                    "latents_out": latents_out,
                },
                os.path.join(self.out_dir, f"step_{index:02d}.pt"),
            )


def _all_tp_ranks(flag: bool) -> bool:
    """True only if every rank says so (host-side, over the world gloo group).

    The step cache picks a different graph per step, and graphs with collectives must
    run on every rank together. The ranks compute bit-identical diffs, so this is a
    guard against a disagreement turning into a hang, not a vote that changes anything.
    """
    if not torch.distributed.is_initialized():
        return flag
    group = get_world_group()
    if group.world_size == 1:
        return flag
    value = torch.tensor([1 if flag else 0], dtype=torch.int32)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
    return bool(value.item())


class _StepCache:
    """Cache-DiT's DBCache (``Fn_compute_blocks=1``, ``Bn_compute_blocks=0``) on the host loop.

    Every step runs the patch embed and the first layer. If the first layer's residual
    ``R = H_1 - H_0`` is within ``residual_diff_threshold`` of the last fully computed
    step's (``mean|R_prev - R| / mean|R_prev|``), the other layers are skipped and their
    residual from that step is added back instead; otherwise they run and both residuals
    are kept. Never in the first ``max_warmup_steps``, never more than
    ``max_continuous_cached_steps`` in a row, at most ``max_cached_steps`` (-1: no limit)
    per request -- the same rules, names and defaults as vllm-omni's Cache-DiT config for
    this model on GPU (``vllm_omni.diffusion.cache.cachedit.config.CacheDiTConfig``).
    """

    DEFAULTS = {
        "residual_diff_threshold": 0.24,
        "max_warmup_steps": 4,
        "max_continuous_cached_steps": 3,
        "max_cached_steps": -1,
    }

    @classmethod
    def parse(cls, value):
        """``model_config.step_cache`` -> keyword arguments, or None when disabled."""
        if not value:
            return None
        if value is True:
            return dict(cls.DEFAULTS)
        if not isinstance(value, dict):
            raise ValueError(f"step_cache must be a bool or a mapping, got {value!r}")
        unknown = set(value) - set(cls.DEFAULTS) - {"enabled"}
        if unknown:
            raise ValueError(f"unknown step_cache keys {sorted(unknown)}; valid: {sorted(cls.DEFAULTS)}")
        if not value.get("enabled", True):
            return None
        return {key: type(default)(value.get(key, default)) for key, default in cls.DEFAULTS.items()}

    def __init__(
        self,
        residual_diff_threshold: float,
        max_warmup_steps: int,
        max_continuous_cached_steps: int,
        max_cached_steps: int,
    ) -> None:
        self.threshold = residual_diff_threshold
        self.warmup = max_warmup_steps
        self.max_continuous = max_continuous_cached_steps
        self.max_cached = max_cached_steps
        self.residual = None  # first layer's residual at the last full step
        self.block_residual = None  # the other layers' residual at that step
        self.continuous = 0
        self.cached_steps: list[int] = []
        self.diffs: list[float] = []

    def wants_cache(self, index: int, diff: float) -> bool:
        if index < self.warmup or self.block_residual is None:
            return False
        if 0 <= self.max_cached <= len(self.cached_steps):
            return False
        if 0 <= self.max_continuous <= self.continuous:
            return False
        return diff < self.threshold  # False for NaN

    def record(self, index: int, diff: float, cached: bool) -> None:
        self.diffs.append(diff)
        if not cached:
            self.continuous = 0
            return
        consecutive = bool(self.cached_steps) and self.cached_steps[-1] == index - 1
        self.continuous = self.continuous + 1 if consecutive else 1
        self.cached_steps.append(index)

    def update(self, residual: torch.Tensor, block_residual: torch.Tensor) -> None:
        self.residual, self.block_residual = residual, block_residual

    def stats(self) -> dict:
        return {
            "cached": len(self.cached_steps),
            "steps": len(self.diffs),
            "cached_steps": list(self.cached_steps),
            "threshold": self.threshold,
            "diffs": [round(d, 4) for d in self.diffs],
        }


class _TeaCache:
    """vllm-omni's TeaCache for HunyuanImage3, on the host-driven denoise loop.

    The same decision as the upstream GPU pipeline's denoise loop
    (``HunyuanImage3Pipeline`` in ``vllm_omni/diffusion/models/hunyuan_image3/
    hunyuan_image3_transformer.py``): the relative L1 change of ``time_embed(t)`` from
    the previous step, rescaled by the model's calibrated polynomial, accumulates until
    it reaches ``rel_l1_thresh``; steps below it reuse the last computed prediction and
    run nothing on the device, and the step that crosses it computes and resets the sum.
    Coefficients and the default threshold come from vllm-omni's ``TeaCacheConfig`` for
    ``HunyuanImage3Pipeline``, so they track upstream.

    The decision depends only on the timesteps, never on the latents, so it is planned
    for the whole request before the first step.
    """

    @classmethod
    def parse(cls, value):
        """``model_config.tea_cache`` -> keyword arguments, or None when disabled."""
        if not value:
            return None
        if value is True:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"tea_cache must be a bool or a mapping, got {value!r}")
        unknown = set(value) - {"enabled", "rel_l1_thresh", "coefficients"}
        if unknown:
            raise ValueError(f"unknown tea_cache keys {sorted(unknown)}")
        if not value.get("enabled", True):
            return None
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig

        # Only pass what was set: older vllm-omni releases reject None for either.
        overrides = {k: value[k] for k in ("rel_l1_thresh", "coefficients") if value.get(k) is not None}
        config = TeaCacheConfig(transformer_type="HunyuanImage3Pipeline", **overrides)
        return {"rel_l1_thresh": float(config.rel_l1_thresh), "coefficients": list(config.coefficients)}

    def __init__(self, rel_l1_thresh: float, coefficients: list[float]) -> None:
        self.threshold = rel_l1_thresh
        self.coefficients = coefficients
        self.compute: list[bool] = []
        self.distances: list[float] = []

    def plan(self, time_embed: nn.Module, timesteps: torch.Tensor) -> list[bool]:
        """Decide every step up front; identical on all TP ranks by construction."""
        import numpy as np

        rescale = np.poly1d(self.coefficients)
        accumulated, previous = 0.0, None
        compute, distances = [], []
        with torch.no_grad():
            for index, timestep in enumerate(timesteps):
                current = time_embed(timestep.unsqueeze(0))
                should_compute = True
                if index > 0 and previous is not None:
                    distance = (
                        (current - previous).abs().mean() / (previous.abs().mean() + 1e-8)
                    ).item()
                    accumulated += abs(float(rescale(distance)))
                    distances.append(round(distance, 4))
                    if accumulated < self.threshold:
                        should_compute = False
                    else:
                        accumulated = 0.0
                previous = current
                compute.append(should_compute)
        # Every rank must run the same graphs: compute a step if any rank would.
        flags = torch.tensor([int(c) for c in compute], dtype=torch.int32)
        if torch.distributed.is_initialized():
            group = get_world_group()
            if group.world_size > 1:
                torch.distributed.all_reduce(
                    flags, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
                )
        self.compute = [bool(f) for f in flags.tolist()]
        self.distances = distances
        return self.compute

    def stats(self) -> dict:
        skipped = [i for i, c in enumerate(self.compute) if not c]
        return {
            "computed": len(self.compute) - len(skipped),
            "steps": len(self.compute),
            "skipped_steps": skipped,
            "rel_l1_thresh": self.threshold,
        }


def _broadcast_from_rank0(value):
    """``value`` as rank 0 has it, on every rank (host-side, over the world gloo group)."""
    holder = [value]
    torch.distributed.broadcast_object_list(holder, src=0, group=get_world_group().cpu_group)
    return holder[0]


def _prompt_buckets(model_config: dict) -> list[int]:
    """The prompt buckets, ascending: ``prefill_len`` and every ``prefill_buckets`` entry."""
    prefill_len = int(model_config.get("prefill_len", default_prefill_len()))
    return sorted({prefill_len, *(int(b) for b in model_config.get("prefill_buckets") or [])})
