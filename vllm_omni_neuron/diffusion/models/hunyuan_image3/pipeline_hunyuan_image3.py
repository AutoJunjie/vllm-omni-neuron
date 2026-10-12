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
from vllm_omni.diffusion.distributed.utils import get_local_device
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
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import (
    TokenizerWrapper,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
    get_hunyuan_image_3_pre_process_func,  # noqa: F401 - re-exported for the registry
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    NeuronHunyuanImage3Transformer,
    default_prefill_len,
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
    support_image_input: ClassVar[bool] = False
    _dit_modules: ClassVar[list[str]] = ["model"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = []

    def __init__(self, od_config, prefix: str = "") -> None:
        del prefix
        self.hf_config = get_config(od_config.model, trust_remote_code=True)
        # PreTrainedModel.__init__ only records config/attributes — no device work.
        HunyuanImage3PreTrainedModel.__init__(self, self.hf_config)
        self.generation_config = GenerationConfig.from_pretrained(od_config.model)
        self.od_config = od_config
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
        self.prefill_len = int(model_config.get("prefill_len", default_prefill_len()))
        self.vae_dtype = getattr(torch, str(model_config.get("vae_dtype", "float32")))
        # Where the output rank decodes. `cpu` is the float32 reference; `neuron` puts the
        # decoder on this rank's core in 16 bits (vae_device_dtype) with fp32 reductions.
        self.vae_device = str(model_config.get("vae_device", "cpu"))
        if self.vae_device not in ("cpu", "neuron"):
            raise ValueError(f"vae_device must be 'cpu' or 'neuron', got {self.vae_device!r}")
        use_nki_mlp = nki_mlp_enabled(model_config)
        use_nki_attention = nki_attention_enabled(model_config)
        moe_block_size = int(model_config.get("moe_block_size", 512))
        moe_implementation = str(model_config.get("moe_implementation", "shard_on_i_hybrid"))
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
        )

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

        self._tkwrapper = TokenizerWrapper(od_config.model)
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
        from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import (
            AutoencoderKLConv3D,
        )

        return AutoencoderKLConv3D.from_config(config.vae)

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
        model_path = self.od_config.model
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_path)
        return model_path

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
        if self.unet_parallel and self.model.tp_size > 1:
            self._unet_shard = UNetShard(
                self.patch_embed, self.final_layer, self.model.tp_rank, self.model.tp_size
            )
            self._unet_shard.group = self.model.tp_group
            logger.info(
                "HunyuanImage3: patch embed / final layer split across %d TP ranks",
                self.model.tp_size,
            )

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

    def _embed_step_inputs(self, latents: torch.Tensor, timestep: torch.Tensor):
        """Patch embed: ``[timestep token] + image tokens``, plus the token grid size."""
        t_emb = self.time_embed(timestep)
        shard = self._unet_shard
        if shard is not None:
            image_tokens, token_h, token_w = unet_down_tokens(shard, latents, t_emb, shard.all_gather)
        else:
            image_tokens, token_h, token_w = _unet_down_tokens(self.patch_embed, latents, t_emb)
        timestep_token = self.timestep_emb(timestep).reshape(latents.shape[0], 1, -1)
        return torch.cat((timestep_token, image_tokens), dim=1), token_h, token_w

    def _unembed_step_output(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor, token_h: int, token_w: int
    ) -> torch.Tensor:
        """Final layer: the image tokens back to a latent-shaped prediction."""
        shard = self._unet_shard
        t_emb = self.time_embed_2(timestep)
        if shard is not None:
            return unet_up_image(
                shard, hidden_states[:, 1:, :], t_emb, token_h, token_w, shard.all_gather
            )
        return _unet_up_image(self.final_layer, hidden_states[:, 1:, :], t_emb, token_h, token_w)

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
        hidden_states, token_h, token_w = self._embed_step_inputs(latents, timestep)
        hidden_states = self.model.forward_denoise(
            hidden_states,
            cos,
            sin,
            (timestep_key_bias, image_key_bias),
            *prompt_kv,
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
        the same (all-reduced) hidden states, so every rank computes the same diff.
        """
        hidden_in, _, _ = self._embed_step_inputs(latents, timestep)
        hidden = self.model.layers[0].forward_denoise(
            hidden_in, cos, sin, (timestep_key_bias, image_key_bias), prompt_key, prompt_value
        )
        residual = hidden - hidden_in
        previous = previous_residual.float()
        diff = (previous - residual.float()).abs().mean() / previous.abs().mean()
        return hidden, residual, diff.reshape(1)

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
        for index, layer in enumerate(self.model.layers[1:]):
            hidden = layer.forward_denoise(
                hidden,
                cos,
                sin,
                (timestep_key_bias, image_key_bias),
                prompt_kv[2 * index],
                prompt_kv[2 * index + 1],
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
            shape = (latents.shape[0], 1 + token_h * token_w, int(self.hf_config.hidden_size))
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

    def _key_biases(self, attention_mask: torch.Tensor, prompt_len: int, image_len: int):
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
        if not bool(torch.all(image_row[:prompt_len])) or not bool(
            torch.all(timestep_row[:prompt_len])
        ):
            raise ValueError(
                "The Neuron HunyuanImage3 path requires generated tokens to attend to "
                "the whole prompt prefix."
            )

        def build(row: torch.Tensor) -> torch.Tensor:
            bias = torch.full(
                (1, 1, 1, self.prefill_len + image_len), _MASK_BIAS, dtype=torch.float32
            )
            bias[..., :prompt_len] = 0.0
            bias[0, 0, 0, self.prefill_len :] = torch.where(
                row[prompt_len:],
                torch.zeros((), dtype=torch.float32),
                torch.full((), _MASK_BIAS, dtype=torch.float32),
            )
            return bias.to(self.device)

        return build(timestep_row), build(image_row)

    def _prompt_inputs(self, input_ids: torch.Tensor, cos, sin, prompt_len: int):
        """Right-pad the prompt to ``prefill_len`` and build its embeddings and RoPE.

        Right-padding is safe because the prefill attends causally: the padded tail
        cannot influence any real prompt position. The pad columns are then masked out
        of every denoise step by the key biases.
        """
        if prompt_len > self.prefill_len:
            raise ValueError(
                f"Prompt needs {prompt_len} tokens but prefill_len={self.prefill_len}. "
                "Raise engine_args.model_config.prefill_len (a longer bucket costs a "
                "one-off recompile)."
            )
        bsz = input_ids.shape[0]
        pad_id = int(getattr(self.hf_config, "pad_token_id", 0) or 0)
        padded_ids = torch.full((bsz, self.prefill_len), pad_id, dtype=torch.long)
        padded_ids[:, :prompt_len] = input_ids[:, :prompt_len].cpu()
        embeds = self.wte(padded_ids).to(dtype=self.model.dtype)

        pad_cos = torch.zeros((bsz, self.prefill_len, cos.shape[-1]), dtype=torch.float32)
        pad_sin = torch.zeros((bsz, self.prefill_len, sin.shape[-1]), dtype=torch.float32)
        pad_cos[:, :prompt_len] = cos[:, :prompt_len].float().cpu()
        pad_sin[:, :prompt_len] = sin[:, :prompt_len].float().cpu()
        return (
            embeds.to(self.device),
            pad_cos.to(self.device),
            pad_sin.to(self.device),
        )

    def _denoise_rope(self, cos, sin, image_mask: torch.Tensor, prompt_len: int):
        """RoPE for ``[timestep token] + image tokens``, gathered by absolute position."""
        bsz, seq_len = image_mask.shape
        index = torch.arange(seq_len, device=image_mask.device).unsqueeze(0).repeat(bsz, 1)
        image_positions = index.masked_select(image_mask.bool()).reshape(bsz, -1)
        timestep_positions = torch.full(
            (bsz, 1), prompt_len, dtype=image_positions.dtype, device=image_positions.device
        )
        positions = torch.cat((timestep_positions, image_positions), dim=1)
        step_cos = real_batched_index_select(cos, 1, positions).float()
        step_sin = real_batched_index_select(sin, 1, positions).float()
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
            req.prompts, extra_args, request_id=req.request_id, allow_cond_image=False
        )
        if batch_cond_image_info:
            raise ValueError(
                "The Neuron HunyuanImage3 pipeline implements text-to-image only; "
                "conditioning images are not supported."
            )
        prompt = prompt_from_req or prompt
        from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_ar import (
            AR_BOT_TASKS,
        )

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
        if bot_task in AR_BOT_TASKS and not any(text is not None for text in cot_text_list):
            # Think / recaption on the same backbone first; the DiT then conditions on
            # the text, as upstream's ar2diffusion hands it over (bot_task back to auto).
            text_prompt = prompt[0] if isinstance(prompt, list) else prompt
            self._ar_stats = self._ar_generator().generate(text_prompt, bot_task)
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

        generator = sampling.generator or generator
        if generator is None:
            # Keep RNG on the host: torch.Generator(<neuron device>) is unsupported, and
            # a CPU generator reproduces the diffusers/GPU trajectory for a given seed.
            seeds = self.prepare_seed(seed=getattr(sampling, "seed", None), batch_size=1)
            generator = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]

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

        attention_mask = self._prepare_attention_mask_for_generation(
            input_ids, self.generation_config, model_kwargs=model_inputs
        )
        # The denoise step covers the timestep token plus the <img> tokens, which is
        # what `image_mask` marks; the template's trailing <eoi>/suffix tokens are not
        # part of the step (upstream's steady-state step drops them too).
        image_len = 1 + int(image_mask[0].sum().item())
        timestep_key_bias, image_key_bias = self._key_biases(
            attention_mask, prompt_len, image_len
        )

        cos, sin = self.get_pos_emb(model_inputs["custom_pos_emb"], model_inputs["position_ids"])

        prefill_start = time.perf_counter()
        prompt_embeds, prompt_cos, prompt_sin = self._prompt_inputs(
            input_ids, cos, sin, prompt_len
        )
        prompt_kv = self.model.run_prefill(prompt_embeds, prompt_cos, prompt_sin)
        prefill_seconds = time.perf_counter() - prefill_start

        step_cos, step_sin = self._denoise_rope(cos, sin, image_mask, prompt_len)

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
        cfg_factor = 1 + int(guidance > 1.0)
        if cfg_factor != input_ids.shape[0]:
            raise ValueError(
                f"CFG factor {cfg_factor} does not match the templated batch "
                f"{input_ids.shape[0]}"
            )

        trace = _StepTrace.from_env(self.is_output_rank)
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
            model_input = latents.repeat(cfg_factor, 1, 1, 1).to(self.model.dtype)
            model_input = model_input.to(self.device)
            t_expand = timestep.repeat(cfg_factor).to(torch.float32).to(self.device)
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
            if trace is not None:
                trace.record(index, timestep, latents, raw_pred, pred)
            latents = self.scheduler.step(pred, timestep, latents, return_dict=False)[0]
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
            if core is not None:
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
            if image is None:
                host_latents = latents.to(dtype=self.vae_dtype)
                image = self.vae.decode(host_latents, return_dict=False, generator=generator)[0]
        if temporal:
            image = image.squeeze(2)

        processor = VaeImageProcessor(
            vae_scale_factor=int(self.hf_config.vae_downsample_factor[0])
        )
        images = processor.postprocess(
            image.to(torch.float32), output_type="pil", do_denormalize=[True] * image.shape[0]
        )
        return images[0]

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
    output rank saves each step's input latents, the raw CFG-batch prediction and the
    combined one. ``HY3_TRACE_FORCE_DIR``: every step starts from the latents a previous
    trace recorded there instead of its own -- teacher forcing, so two implementations are
    compared on identical inputs at every step rather than on diverging trajectories.
    ``HY3_TRACE_STEPS`` (``0,25,49``): with forcing, compute only those steps.
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
        if steps is not None and not force_dir:
            raise ValueError("HY3_TRACE_STEPS needs HY3_TRACE_FORCE_DIR")
        logger.warning(
            "HunyuanImage3 step trace: out=%s force=%s steps=%s", out_dir, force_dir, steps
        )
        return cls(out_dir, force_dir, steps, is_output_rank)

    def done(self, index):
        return self.steps is not None and index > self.steps[-1]

    def skip(self, index):
        return self.steps is not None and index not in self.steps

    def forced_latents(self, index, latents):
        if not self.force_dir:
            return latents
        return torch.load(os.path.join(self.force_dir, f"step_{index:02d}.pt"))["latents_in"]

    def record(self, index, timestep, latents, raw_pred, pred):
        if self.out_dir and self.save:
            torch.save(
                {
                    "timestep": timestep.clone(),
                    "latents_in": latents.clone(),
                    "pred_raw": raw_pred.clone(),
                    "pred": pred.clone(),
                },
                os.path.join(self.out_dir, f"step_{index:02d}.pt"),
            )


def _all_tp_ranks(flag: bool) -> bool:
    """True only if every TP rank says so (host-side, over the gloo group).

    The step cache picks a different graph per step, and graphs with collectives must
    run on every rank together. The ranks compute bit-identical diffs, so this is a
    guard against a disagreement turning into a hang, not a vote that changes anything.
    """
    if not torch.distributed.is_initialized():
        return flag
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
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
            from vllm.distributed.parallel_state import get_tp_group

            group = get_tp_group()
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
