# SPDX-License-Identifier: Apache-2.0
"""NeuronHunyuanImage3Pipeline — HunyuanImage-3.0 text-to-image for Neuron.

Subclasses vLLM-Omni's ``HunyuanImage3Pipeline`` and keeps all of its pure-Python
request handling — chat templating, 2D-RoPE construction, the generation attention
mask, image-info/resolution bucketing — while replacing everything from ``forward()``
down with the Neuron execution path:

* the 32-layer MoE backbone becomes :class:`NeuronHunyuanImage3Transformer`
  (head-parallel attention + expert-parallel MoE, NKI kernels, raw ``nn.Parameter``
  weights), optionally as two TP replicas that run one CFG branch each;
* the DiT runs as **two** fixed-shape ``torch.compile`` graphs — a prompt prefill that
  emits the per-layer prompt K/V, and a denoise step that fuses the patch embedder, the
  backbone and the final layer into one NEFF replayed every scheduler step;
* the scheduler, the CFG combine and the token embedding stay on the host, so no
  uncompiled op is ever dispatched to the device under the Lite runtime;
* the VAE decode runs on the host in float32, or with ``vae_device: neuron`` as seven
  staged float16 graphs on the output rank's core or split by rows over a few ranks
  (:mod:`.vae_device`).

This is the DiT-only deployment: the prompt goes straight to the diffusion stage with
no autoregressive think / recaption stage (``bot_task: image``), matching upstream's
``vllm_omni/deploy/hunyuan_image3_dit.yaml``. A request is text in, one 1024 x 1024
image out, with classifier-free guidance; :meth:`NeuronHunyuanImage3Pipeline._check_request`
rejects what this build does not run.
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
    cfg_gather_batch,
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

# What a request may ask for in this build: text-to-image with no think / recaption text
# (upstream's single-stage bot_task "image"; "auto" and an unset task template the same
# tokens for a gen_image request), at the one compiled resolution.
SUPPORTED_BOT_TASKS = (None, "auto", "image", "vanilla")
SUPPORTED_SIZE = (1024, 1024)

# model_config keys of features this build does not include. Set to anything but a false
# value, they raise instead of being silently ignored.
UNSUPPORTED_MODEL_CONFIG = {
    "step_cache": "the Cache-DiT step cache",
    "tea_cache": "TeaCache",
    "prefill_buckets": "more than one prompt bucket",
    "cfg_combine": "NeuronCFGParallelMixin's CFG combine",
    "tp_sequence_parallel": "sequence parallelism inside TP",
    "enable_ring_attention": "ring attention",
    "vae_device_core": "the VAE decode on a separate core",
    "vae_encode_device": "conditioning images (image editing)",
    "ar_max_new_tokens": "think / recaption on the backbone",
}


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
    # Text-to-image only: no conditioning images.
    support_image_input: ClassVar[bool] = False
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
        _check_model_config(model_config)
        parallel_config = getattr(od_config, "parallel_config", None)
        if int(getattr(parallel_config, "vae_patch_parallel_size", 1) or 1) > 1:
            raise NotImplementedError(
                "HunyuanImage3 on Neuron: parallel_config.vae_patch_parallel_size > 1 (vLLM-Omni's "
                "tile-parallel VAE) is not supported in this build; use model_config.vae_parallel"
            )
        # The prompt bucket: the prefill graph is compiled for this length and every
        # prompt is right-padded to it.
        self.prefill_len = int(model_config.get("prefill_len", default_prefill_len()))
        self.vae_dtype = getattr(torch, str(model_config.get("vae_dtype", "float32")))
        # Where the output rank decodes. `cpu` is the float32 reference; `neuron` puts the
        # decoder on device in 16 bits (vae_device_dtype) with fp32 reductions.
        self.vae_device = str(model_config.get("vae_device", "cpu"))
        if self.vae_device not in ("cpu", "neuron"):
            raise ValueError(f"vae_device must be 'cpu' or 'neuron', got {self.vae_device!r}")
        use_nki_mlp = nki_mlp_enabled(model_config)
        use_nki_attention = nki_attention_enabled(model_config)
        use_denoise_kernel = use_nki_attention and denoise_attention_kernel_enabled(model_config)
        moe_block_size = int(model_config.get("moe_block_size", 512))
        moe_implementation = str(model_config.get("moe_implementation", "shard_on_i_hybrid"))
        # Split the patch embed and final layer by GroupNorm group across the ranks
        # (unet_parallel.py) instead of running them whole on every rank.
        self.unet_parallel = bool(model_config.get("unet_parallel", True))
        self._unet_shard = None

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

        # Parallelism comes from parallel_config: tensor_parallel_size, and
        # cfg_parallel_size: 2 for two TP replicas that run one CFG branch each.
        self.model = NeuronHunyuanImage3Transformer(
            config,
            use_nki_mlp=use_nki_mlp,
            use_nki_attention=use_nki_attention,
            moe_block_size=moe_block_size,
            moe_implementation=moe_implementation,
            use_denoise_attention_kernel=use_denoise_kernel,
        )
        self.cfg_parallel = self.model.cfg_size > 1

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
        # replicated on every core), and the VAE decodes once per request.
        # With vae_device=neuron the decoding ranks copy the decoder onto their cores on
        # first use.
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
        model = self.model
        if self.unet_parallel and model.world_size > 1:
            # Over every rank of the stage, whatever the backbone's TP x CFG split: the
            # CFG replicas hold the same latents in front and both branches behind.
            self._unet_shard = UNetShard(
                self.patch_embed, self.final_layer, model.world_rank, model.world_size
            )
            self._unet_shard.group = model.world_group
            logger.info(
                "HunyuanImage3: patch embed / final layer split across %d ranks", model.world_size
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
        """Final layer: the image tokens back to a latent-shaped prediction.

        With CFG replicas each holds one branch; the final layer takes both, as the CFG
        batch, on every rank.
        """
        model = self.model
        if model.cfg_size > 1:
            hidden_states = cfg_gather_batch(hidden_states, model.cfg_size, model.cfg_group)
        t_emb = self.time_embed_2(timestep)
        image_tokens = hidden_states[:, 1 : 1 + token_h * token_w, :]
        shard = self._unet_shard
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
        hidden_states, token_h, token_w = self._embed_step_inputs(latents, timestep)
        hidden_states = self.model.forward_denoise(
            hidden_states, cos, sin, (timestep_key_bias, image_key_bias), *prompt_kv
        )
        return self._unembed_step_output(hidden_states, timestep, token_h, token_w)

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
        self.model.compile(*args, options=options, **kwargs)
        self._compiled_denoise_step = torch.compile(
            self._denoise_step_graph,
            *args,
            options={**options, "model_name": "hunyuan_image3_denoise_step"},
            **kwargs,
        )
        return self

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

        return build(timestep_row), build(image_row)

    def _prompt_inputs(self, input_ids: torch.Tensor, cos, sin, prompt_len: int):
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

    def _request_bot_task(self, req, extra_args: dict):
        """The request's ``bot_task``: its own, else the stage's ``model_config.bot_task``.

        vllm-omni 0.24's diffusion chat endpoint does not forward a request's bot_task
        (only later releases pass model extras through extra_args), so a served stage
        sets one for every request in model_config.
        """
        bot_task = extra_args.get("bot_task")
        first = req.prompts[0] if req.prompts else None
        if bot_task is None and isinstance(first, dict):
            bot_task = first.get("bot_task")
        if bot_task is None:
            bot_task = dict(self.od_config.model_config or {}).get("bot_task")
        return bot_task

    def _check_request(
        self, *, bot_task, cot_text_list, height, width, guidance, cond_images=None
    ) -> None:
        """Reject what this text-to-image build does not run, before any device work."""
        if isinstance(bot_task, str) and bot_task.lower() == "none":
            bot_task = None
        if bot_task not in SUPPORTED_BOT_TASKS:
            raise NotImplementedError(
                f"HunyuanImage3 on Neuron: bot_task {bot_task!r} is not supported; this build "
                "runs text-to-image only (bot_task 'image', no think / recaption)"
            )
        if any(text is not None for text in cot_text_list):
            raise NotImplementedError(
                "HunyuanImage3 on Neuron: think / recaption text from an AR stage is not "
                "supported; this build runs text-to-image only"
            )
        if cond_images and any(cond_images):
            raise NotImplementedError(
                "HunyuanImage3 on Neuron: conditioning images (image editing) are not "
                "supported; this build runs text-to-image only"
            )
        if (int(height), int(width)) != SUPPORTED_SIZE:
            raise ValueError(
                f"HunyuanImage3 on Neuron: {height}x{width} is not supported; this build "
                f"generates {SUPPORTED_SIZE[0]}x{SUPPORTED_SIZE[1]} only"
            )
        if self.cfg_parallel and guidance <= 1.0:
            raise ValueError(
                f"HunyuanImage3 on Neuron: guidance_scale {guidance} turns CFG off, but the "
                "stage runs cfg_parallel_size: 2 (one CFG branch per replica); use a "
                "guidance_scale above 1"
            )

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
        from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID

        if self.skip_warmup and getattr(req, "request_ids", None) == [DUMMY_DIFFUSION_REQUEST_ID]:
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
        generator = sampling.generator or generator
        if generator is None:
            # Keep RNG on the host: torch.Generator(<neuron device>) is unsupported, and
            # a CPU generator reproduces the diffusers/GPU trajectory for a given seed.
            seeds = self.prepare_seed(seed=getattr(sampling, "seed", None), batch_size=1)
            generator = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]

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
        self._check_request(
            bot_task=self._request_bot_task(req, extra_args),
            cot_text_list=cot_text_list,
            cond_images=batch_cond_image_info,
            height=height,
            width=width,
            guidance=guidance,
        )

        model_inputs = self.prepare_model_inputs(
            prompt=prompt,
            cot_text=None,
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
        self._record_perf_metrics(steps=steps, height=height, width=width, e2e=elapsed)
        return DiffusionOutput(output=image)

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
        if prompt_len > self.prefill_len:
            raise ValueError(
                f"Prompt needs {prompt_len} tokens but the prompt bucket is {self.prefill_len}; "
                "raise engine_args.model_config.prefill_len (one more prefill and denoise compile)."
            )

        attention_mask = self._prepare_attention_mask_for_generation(
            input_ids, self.generation_config, model_kwargs=model_inputs
        )
        # The denoise step covers the timestep token plus the <img> tokens, which is
        # what `image_mask` marks; the template's trailing <eoi>/suffix tokens are not
        # part of the step (upstream's steady-state step drops them too).
        image_len = 1 + int(image_mask[0].sum().item())
        cfg_factor = 1 + int(guidance > 1.0)
        if cfg_factor != input_ids.shape[0]:
            raise ValueError(
                f"CFG factor {cfg_factor} does not match the templated batch "
                f"{input_ids.shape[0]}"
            )
        timestep_key_bias, image_key_bias = self._key_biases(attention_mask, prompt_len, image_len)

        cos, sin = self.get_pos_emb(model_inputs["custom_pos_emb"], model_inputs["position_ids"])

        # CFG parallel: this replica's TP group runs CFG branch cfg_rank (0 = cond) whole.
        branch = self.model.cfg_rank if self.cfg_parallel else None

        def rows(tensor):
            # This rank's CFG branch of a batch-major tensor (shared rows stay as they are).
            if branch is None or tensor.shape[0] == 1:
                return tensor
            return tensor[branch : branch + 1]

        prefill_start = time.perf_counter()
        prompt_embeds, prompt_cos, prompt_sin = self._prompt_inputs(
            rows(input_ids), rows(cos), rows(sin), prompt_len
        )
        prompt_kv = self.model.run_prefill(prompt_embeds, prompt_cos, prompt_sin)
        prefill_seconds = time.perf_counter() - prefill_start

        step_cos, step_sin = self._denoise_rope(rows(cos), rows(sin), rows(image_mask), prompt_len)

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

        trace = _StepTrace.from_env(self.is_output_rank)
        if trace is not None:
            trace.record_inputs(
                model_inputs, attention_mask=attention_mask, guidance=guidance, steps=steps
            )
        denoise_start = time.perf_counter()
        for index, timestep in enumerate(timesteps):
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
            pred = (self._compiled_denoise_step or self._denoise_step_graph)(
                model_input,
                t_expand,
                step_cos,
                step_sin,
                timestep_key_bias,
                image_key_bias,
                *prompt_kv,
            )
            # Same rule in the other direction: copy first, then widen on the host.
            pred = pred.cpu().to(torch.float32)
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

        decode_start = time.perf_counter()
        # Only the rank whose output the engine keeps needs pixels. Decoding on all 32
        # ranks is pure duplicated work, and because the Lite worker pins
        # torch.set_num_threads(1) they also contend for the host instead of sharing it.
        vae_parts, vae_rank = self._vae_split()
        if vae_parts > 1:
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
        """Decode latents to a PIL image, on the host or on device. Decoding ranks only."""
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
        """The staged on-device decoder, built on first use. Decoding ranks only.

        Built lazily rather than at load time because only the decoding ranks need it,
        and its graphs compile on the first call for each latent shape anyway. With
        ``vae_parallel: N`` the decode is split by rows over ranks 0..N-1; otherwise it
        runs whole on the output rank's core, next to its backbone shard.
        """
        if self._device_vae_decoder is None:
            # float16 is the default: same speed as bfloat16, 66 dB PSNR to the host
            # decode against 59.
            model_config = dict(self.od_config.model_config or {})
            dtype = str(model_config.get("vae_device_dtype", "float16"))
            if dtype not in ("float16", "bfloat16"):
                raise ValueError(f"vae_device_dtype must be float16 or bfloat16, got {dtype!r}")
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
                    # A failed device decode (device OOM) costs this request the host
                    # decode's 29 s, not the image; the next request rebuilds.
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

    def _vae_split(self) -> tuple[int, int | None]:
        """``(parts, this rank's part or None)`` for ``model_config.vae_parallel``.

        ``vae_parallel: N`` splits the device decode by image rows over ranks 0..N-1 --
        neighbouring cores of the first chip -- instead of running it whole on the output
        rank's core, which at TP16 x CFG2 no longer fits beside a backbone shard.
        The groups are built on first use, by every rank (``new_group`` is collective).
        """
        if getattr(self, "_vae_split_cache", None) is None:
            model_config = dict(self.od_config.model_config or {})
            parts = int(model_config.get("vae_parallel", 1))
            if parts > 1 and self.vae_device != "neuron":
                # A split of the device decode; the host decode runs whole on rank 0.
                logger.info("HunyuanImage3: vae_device=%s, so vae_parallel is unused", self.vae_device)
                parts = 1
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

    Off unless ``HY3_TRACE_DIR`` is set: the output rank then saves the request's inputs
    and each step's input latents, the raw CFG-batch prediction, the combined one and the
    scheduler's updated latents, which a reference replays (``hy3_official_step.py``).
    ``HY3_TRACE_STEPS`` (``0,25,49``): record only those steps.
    """

    def __init__(self, out_dir, steps, save):
        self.out_dir, self.steps, self.save = out_dir, steps, save
        if save:
            os.makedirs(out_dir, exist_ok=True)

    @classmethod
    def from_env(cls, is_output_rank):
        out_dir = os.environ.get("HY3_TRACE_DIR")
        if not out_dir:
            return None
        raw = os.environ.get("HY3_TRACE_STEPS")
        steps = sorted({int(x) for x in raw.split(",")}) if raw else None
        logger.warning("HunyuanImage3 step trace: out=%s steps=%s", out_dir, steps)
        return cls(out_dir, steps, is_output_rank)

    def record_inputs(self, model_inputs, **settings):
        """``inputs.pt``: what a reference needs to replay the request's steps.

        The templated inputs the official implementation's forward takes, plus
        ``settings`` (the attention mask, guidance, steps).
        """
        if self.save:
            keys = ("input_ids", "image_mask", "position_ids", "gen_timestep_scatter_index")
            kept = {key: model_inputs.get(key) for key in keys}
            torch.save({**kept, **settings}, os.path.join(self.out_dir, "inputs.pt"))

    def record(self, index, timestep, latents, raw_pred, pred, latents_out):
        if self.save and (self.steps is None or index in self.steps):
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


def _check_model_config(model_config: dict) -> None:
    """Raise on model_config keys of features this build does not include."""
    for key, feature in UNSUPPORTED_MODEL_CONFIG.items():
        if model_config.get(key):
            raise NotImplementedError(
                f"HunyuanImage3 on Neuron: model_config.{key} ({feature}) is not supported "
                "in this build; remove it from the stage config"
            )
