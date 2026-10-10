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
* the scheduler, the CFG combine, the token embedding and the VAE decode stay on the
  host, so no uncompiled op is ever dispatched to the device under the Lite runtime.

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
        use_nki_mlp = nki_mlp_enabled(model_config)
        use_nki_attention = nki_attention_enabled(model_config)
        moe_block_size = int(model_config.get("moe_block_size", 512))

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

        # The token embedding and the VAE stay on the host: the embedding is a single
        # gather over a 133k x 4096 table (1.1 GB that would otherwise be replicated on
        # every core), and the VAE decodes once per request.
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
            getattr(self, name).to(*args, **kwargs)
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
        t_emb = self.time_embed(timestep)
        image_tokens, token_h, token_w = _unet_down_tokens(self.patch_embed, latents, t_emb)
        timestep_token = self.timestep_emb(timestep).reshape(latents.shape[0], 1, -1)
        hidden_states = torch.cat((timestep_token, image_tokens), dim=1)

        hidden_states = self.model.forward_denoise(
            hidden_states,
            cos,
            sin,
            (timestep_key_bias, image_key_bias),
            *prompt_kv,
        )
        return _unet_up_image(
            self.final_layer,
            hidden_states[:, 1:, :],
            self.time_embed_2(timestep),
            token_h,
            token_w,
        )

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

    def _run_denoise_step(self, *call_args):
        fn = self._compiled_denoise_step or self._denoise_step_graph
        return fn(*call_args)

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

        denoise_start = time.perf_counter()
        for index, timestep in enumerate(timesteps):
            self._current_timestep = timestep
            # Cast on the host and transfer separately: the Lite runtime's copy
            # requires matching dtypes, so a single .to(device=..., dtype=...) raises
            # "Expected self.dtype() == dst.dtype()".
            model_input = latents.repeat(cfg_factor, 1, 1, 1).to(self.model.dtype)
            model_input = model_input.to(self.device)
            t_expand = timestep.repeat(cfg_factor).to(torch.float32).to(self.device)
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
            pred = pred.cpu().to(torch.float32)
            if cfg_factor == 2:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + guidance * (pred_cond - pred_uncond)
            latents = self.scheduler.step(pred, timestep, latents, return_dict=False)[0]
            if index == 0:
                logger.info("HunyuanImage3: first denoise step complete (graph warm)")
        denoise_seconds = time.perf_counter() - denoise_start
        self._current_timestep = None

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
        """Decode latents to a PIL image on the host. Output rank only."""
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

    def _decode_impl(self, latents: torch.Tensor, generator, VaeImageProcessor):
        vae_config = self.vae.config
        latents = latents.to(dtype=self.vae_dtype)
        if getattr(vae_config, "scaling_factor", None):
            latents = latents / vae_config.scaling_factor
        if getattr(vae_config, "shift_factor", None):
            latents = latents + vae_config.shift_factor
        temporal = hasattr(self.vae, "ffactor_temporal")
        if temporal:
            latents = latents.unsqueeze(2)
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
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
