# SPDX-License-Identifier: Apache-2.0
"""Worker for the tier-1 component tests: one decoder layer, the UNet ends, the VAE decoder.

Run by ``test_hunyuan_image3_components_accuracy.py``, one process per measurement:

- ``--mode backbone --device neuron``: under ``torchrun --nproc-per-node 16``, decoder
  layer 0 of the checkpoint at the default config's TP16 sharding (NKI attention, o-proj
  and MoE kernels, compiled), its prompt prefill and one denoise step; then the patch
  embed and final layer split over the 16 ranks (unet_parallel). Rank 0 saves.
- ``--mode backbone --device official --dtype {float32,bfloat16}``: the reference --
  the official implementation's decoder layer 0, patch embed and final layer on the
  host (:mod:`hy3_official`), over the same tokens and attention pattern.
- ``--mode vae --device neuron [--vae-parallel 2]``: the staged device decoder
  (float16, float32 GroupNorm reductions) on one core, or -- under ``torchrun`` with the
  stage's world -- split by rows over ranks 0 and 1 as the default stage config runs it.
- ``--mode vae --device official --dtype {float32,float16}``: the official VAE decode
  on the host; in float16 with float32 GroupNorm statistics, the device's arithmetic.

The inputs (``--inputs``, written by the test) are shared, so the runs differ only in
implementation, device and dtype.
"""

from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["backbone", "vae"], required=True)
parser.add_argument("--device", choices=["neuron", "official"], required=True)
parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
parser.add_argument("--vae-parallel", type=int, default=1)
parser.add_argument("--model-path", required=True)
parser.add_argument("--inputs", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

if args.device == "neuron":
    if "LOCAL_RANK" in os.environ:
        # One visible core per torchrun process and the CCOM root, as the diffusion worker sets.
        os.environ["NEURON_RT_VISIBLE_CORES"] = os.environ["LOCAL_RANK"]
        os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61234"
        os.environ.pop("NEURON_LIBRARY_PATH", None)
        os.environ.setdefault("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")
        os.environ.setdefault("NEURON_RT_DBG_CC_DMA_PACKET_SIZE", "2048")
        os.environ.setdefault("NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE", "167772160")
        os.environ["OMP_NUM_THREADS"] = "1"
    for key, value in {
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "VLLM_NEURON_BACKEND": "neuron_native",
        "VLLM_NEURON_LIBTORCH_NEURONX_LITE": "1",
        "VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND": "1",
        "NEURON_SCRATCHPAD_PAGE_SIZE": "2048",
        "NEURON_RT_EXEC_TIMEOUT": "600",
    }.items():
        os.environ.setdefault(key, value)
    import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from hy3_official import read  # noqa: E402

COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1", "--hbm-scratchpad-page-size=2048"]
PROMPT = 300  # the live prompt rows of the inputs' 512-row bucket


def _init_world() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_neuron.vllm.platform import NeuronPlatform
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment

    from vllm_omni_neuron.lite_compat import initialize as initialize_lite

    world = int(os.environ["WORLD_SIZE"])
    NeuronPlatform.set_device_count(world)
    initialize_lite()
    init_distributed_environment(
        world_size=world, rank=int(os.environ["RANK"]), local_rank=int(os.environ["LOCAL_RANK"])
    )
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    _init_world._context = context


def _device():
    from vllm_omni.platforms import current_omni_platform

    return current_omni_platform.get_torch_device(0)


def _compile(fn, name):
    from vllm_neuron.envs import get_compile_backend_name

    return torch.compile(fn, backend=get_compile_backend_name(), fullgraph=True, dynamic=False,
                         options={"model_name": name, "compiler_args": COMPILER_ARGS})


def backbone_neuron() -> None:
    import torch.nn as nn
    from vllm.transformers_utils.config import get_config
    from vllm_omni.diffusion.distributed.parallel_state import initialize_model_parallel
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        TimestepEmbedder,
        UNetDown,
        UNetUp,
    )

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        NeuronHunyuanImage3Transformer,
    )
    from vllm_omni_neuron.diffusion.models.hunyuan_image3.unet_parallel import (
        UNetShard,
        unet_down_tokens,
        unet_up_image,
    )

    torch.nn.functional.gelu = torch.ops.aten.gelu.default
    _init_world()
    initialize_model_parallel(tensor_parallel_size=int(os.environ["WORLD_SIZE"]))
    dtype = torch.bfloat16
    device = _device()
    config = get_config(args.model_path, trust_remote_code=True)
    config.num_hidden_layers = 1
    torch.set_default_dtype(dtype)
    with torch.device("cpu"):
        model = NeuronHunyuanImage3Transformer(config, use_denoise_attention_kernel=True)
    torch.set_default_dtype(torch.float32)
    model.load_weights(args.model_path)
    model.to(device)
    inputs = torch.load(args.inputs)

    def on(t, dt=None):
        return t.to(dt or t.dtype).to(device)

    prompt = (on(inputs["prompt_embeds"]), on(inputs["prompt_cos"]), on(inputs["prompt_sin"]))
    step = (on(inputs["step_hidden"]), on(inputs["step_cos"]), on(inputs["step_sin"]))
    biases = (on(inputs["timestep_key_bias"]), on(inputs["image_key_bias"]))
    with torch.no_grad():
        prompt_kv = _compile(model.forward_prefill, "hy3_tier1_prefill")(*prompt)
        denoise = _compile(lambda h, c, s, tb, ib, *kv: model.forward_denoise(h, c, s, (tb, ib), *kv),
                           "hy3_tier1_denoise")(*step, *biases, *prompt_kv)
    result = {"key": prompt_kv[0].cpu().float(), "value": prompt_kv[1].cpu().float(),
              "denoise": denoise.cpu().float(), "tp_rank": model.tp_rank, "tp_size": model.tp_size}

    # The patch embed and the final layer, split over the stage's ranks (unet_parallel).
    hidden = int(config.hidden_size)
    patch_embed = UNetDown(patch_size=config.patch_size, emb_channels=hidden, in_channels=config.vae["latent_channels"],
                           hidden_channels=config.patch_embed_hidden_dim, out_channels=hidden)
    final_layer = UNetUp(patch_size=config.patch_size, emb_channels=hidden, in_channels=hidden,
                         hidden_channels=config.patch_embed_hidden_dim, out_channels=config.vae["latent_channels"],
                         out_norm=True)
    holder = nn.Module()
    holder.time_embed, holder.time_embed_2 = TimestepEmbedder(hidden_size=hidden), TimestepEmbedder(hidden_size=hidden)
    holder.patch_embed, holder.final_layer = patch_embed, final_layer
    for name, part in holder.named_children():
        part.load_state_dict(read(args.model_path, f"{name}."))
    holder = holder.to(dtype).eval().requires_grad_(False)
    shard = UNetShard(patch_embed, final_layer, model.world_rank, model.world_size)
    shard.group = model.world_group
    shard, holder = shard.to(device), holder.to(device)
    latents, timestep = on(inputs["latents"]), on(inputs["timestep"], torch.float32)
    tokens = on(inputs["unet_tokens"])

    def down(lat, t):
        return unet_down_tokens(shard, lat, holder.time_embed(t), shard.all_gather)[0]

    def up(tok, t):
        return unet_up_image(shard, tok, holder.time_embed_2(t), 64, 64, shard.all_gather)

    with torch.no_grad():
        result["patch_embed"] = _compile(down, "hy3_tier1_patch_embed")(latents, timestep).cpu().float()
        result["final_layer"] = _compile(up, "hy3_tier1_final_layer")(tokens, timestep).cpu().float()
    if dist.get_rank() == 0:
        torch.save(result, args.output)


def backbone_official() -> None:
    """The reference: the official decoder layer over the prompt and step as one sequence.

    The device splits that sequence into a padded prompt prefill and a denoise step over
    its K/V; here it is the live prompt rows followed by the step rows, with the
    attention pattern the inputs' key biases describe written as one additive mask.
    """
    from hy3_official import build, load, official_module

    torch.set_num_threads(os.cpu_count())
    dtype = getattr(torch, args.dtype)
    module, config = official_module(args.model_path)
    model = build(lambda: module.HunyuanImage3ForCausalMM(config), dtype)
    layer = load(model.model.layers[0], args.model_path, "model.layers.0.")
    inputs = torch.load(args.inputs)
    step_len = inputs["step_hidden"].shape[1]
    hidden = torch.cat((inputs["prompt_embeds"][:, :PROMPT], inputs["step_hidden"]), dim=1).to(dtype)
    half_cos = torch.cat((inputs["prompt_cos"][:, :PROMPT], inputs["step_cos"]), dim=1)
    half_sin = torch.cat((inputs["prompt_sin"][:, :PROMPT], inputs["step_sin"]), dim=1)
    cos = torch.cat((half_cos, half_cos), dim=-1).to(dtype)
    sin = torch.cat((half_sin, half_sin), dim=-1).to(dtype)
    total = PROMPT + step_len
    allowed = torch.zeros(total, total, dtype=torch.bool)
    allowed[:PROMPT, :PROMPT] = torch.ones(PROMPT, PROMPT, dtype=torch.bool).tril()
    allowed[PROMPT:, :PROMPT] = True  # the step sees the whole prompt
    allowed[PROMPT, PROMPT] = True  # the timestep token sees itself
    allowed[PROMPT + 1 :, PROMPT:] = True  # image tokens see the whole step
    mask = torch.zeros(1, 1, total, total, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)

    class Capture:
        """A one-layer cache that only records the layer's K/V (post-RoPE, post-norm)."""

        def update(self, key, value, layer_idx, cache_kwargs=None):
            self.key, self.value = key, value
            return key, value

    capture = Capture()
    with torch.no_grad():
        out = layer(hidden, attention_mask=mask, past_key_value=capture, custom_pos_emb=(cos, sin))[0]
    result = {
        "key": capture.key.transpose(1, 2).float(),
        "value": capture.value.transpose(1, 2).float(),
        "denoise": out[:, PROMPT:].float(),
    }

    parts = {}
    for name in ("patch_embed", "final_layer", "time_embed", "time_embed_2"):
        parts[name] = load(getattr(model, name), args.model_path, f"{name}.")
    timestep = inputs["timestep"].float()
    with torch.no_grad():
        result["patch_embed"] = parts["patch_embed"](
            inputs["latents"].to(dtype), parts["time_embed"](timestep)
        )[0].float()
        result["final_layer"] = parts["final_layer"](
            inputs["unet_tokens"].to(dtype), parts["time_embed_2"](timestep), 64, 64
        ).float()
    torch.save(result, args.output)


def vae_neuron() -> None:
    from vllm.transformers_utils.config import get_config
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    from vllm_omni_neuron.lite_compat import ensure_current_device_index
    from vllm_omni_neuron.lite_compat import initialize as initialize_lite

    torch.nn.functional.gelu = torch.ops.aten.gelu.default
    dtype = getattr(torch, args.dtype)

    def host_vae():
        config = get_config(args.model_path, trust_remote_code=True)
        vae = AutoencoderKLConv3D.from_config(dict(config.vae)).eval().requires_grad_(False)
        vae.load_state_dict(read(args.model_path, "vae."), strict=True)
        return vae, dict(config.vae)

    latents = torch.load(args.inputs)["vae_latents"]
    if args.vae_parallel == 1:
        from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import StagedDeviceDecoder

        initialize_lite()
        ensure_current_device_index()
        host, vae_config = host_vae()
        image = StagedDeviceDecoder(host, vae_config, _device(), dtype=dtype).decode(latents.float())
        torch.save({"image": image.cpu().float()}, args.output)
        return

    # The row split with the pipeline's groups (vae_row_groups over the whole world, as
    # the graphs carry them); block 0 decodes.
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (
        ShardedDeviceDecoder,
        bootstrap_device_collectives,
        vae_row_groups,
    )

    _init_world()
    ensure_current_device_index()
    rank, parts = dist.get_rank(), args.vae_parallel
    group, _ = vae_row_groups(parts)
    bootstrap_device_collectives(_device())
    if rank < parts:
        host, vae_config = host_vae()
        image = ShardedDeviceDecoder(host, vae_config, _device(), dtype, rank, parts, group).decode(latents.float())
        if rank == 0:
            torch.save({"image": image.cpu().float()}, args.output)
    dist.all_reduce(torch.zeros(1), group=get_world_group().cpu_group)  # gloo, not dist.barrier


def vae_official() -> None:
    from hy3_official import build, load, official_module

    torch.set_num_threads(os.cpu_count())
    dtype = getattr(torch, args.dtype)
    module, config = official_module(args.model_path)
    vae_cls = module.AutoencoderKLConv3D
    vae = load(build(lambda: vae_cls.from_config(config.vae), dtype), args.model_path, "vae.", strict=False)
    if dtype != torch.float32:
        # The device decoder's arithmetic: 16-bit weights and activations, float32
        # GroupNorm statistics. Written here against torch's GroupNorm, not the plugin's.
        def fp32_group_norm(self, x):
            return torch.nn.functional.group_norm(
                x.float(), self.num_groups, self.weight.float(), self.bias.float(), self.eps
            ).to(x.dtype)

        for norm in (m for m in vae.modules() if isinstance(m, torch.nn.GroupNorm)):
            norm.forward = fp32_group_norm.__get__(norm)
    latents = torch.load(args.inputs)["vae_latents"].to(dtype)
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    # A single-frame latent keeps the last decoded frame, as the pipeline's decode does.
    torch.save({"image": image[:, :, -1:].float()}, args.output)


if __name__ == "__main__":
    {
        ("backbone", "neuron"): backbone_neuron,
        ("backbone", "official"): backbone_official,
        ("vae", "neuron"): vae_neuron,
        ("vae", "official"): vae_official,
    }[(args.mode, args.device)]()
