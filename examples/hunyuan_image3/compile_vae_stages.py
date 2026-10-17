# SPDX-License-Identifier: Apache-2.0
"""Fill the NEFF cache for the pipeline's on-device VAE decode, all stages in parallel.

With ``vae_device: neuron`` the output rank compiles the decoder's seven stage graphs on
the first request at each resolution. Serially that is ~2 hours cold at 1024x1024
(15 / 101 / 755 / 1489 / 2537 / 2094 / 100 s), all of it inside one request. But a stage
graph depends only on its input shape, not on the activation feeding it, so each stage
can compile in its own process from a zero tensor of the right shape -- the wall time is
the slowest stage (~42 min), and the pipeline then hits every NEFF.

Weights are random: the parameters are graph inputs, so their values never reach the
compiler. Shapes come from a pass over the decoder on the meta device.

    python examples/hunyuan_image3/compile_vae_stages.py                 # all, cores 0-6
    python examples/hunyuan_image3/compile_vae_stages.py --stage up3     # one, this core
    python examples/hunyuan_image3/compile_vae_stages.py --vae-parallel 2   # the default

``--vae-parallel N`` fills the cache for the stage configs' row split (``vae_parallel``,
:class:`ShardedDeviceDecoder`) instead. Its stage graphs carry their collectives'
replica groups -- the whole world's ``N``-rank blocks -- so it runs the stage's world
(``--world-size``, 32 for the TP32 configs, one process per core 0..31): block ``b``
compiles stage ``b`` for each of its ranks' bands, the other blocks only take part in
creating the groups.

Run it from the same checkout the server imports: the cache key includes the source path
and line of every traced op (see ``vae_device``).
"""

import argparse
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description="Precompile the staged device VAE decode")
parser.add_argument("--model-path", default="/workspace/models/HunyuanImage-3.0-Instruct")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument(
    "--dtype",
    default="float16",
    choices=["float16", "bfloat16"],
    help="Device dtype of the decoder; match the stage config's vae_device_dtype.",
)
parser.add_argument("--stage", default=None, help="Compile only this stage, on this process")
parser.add_argument(
    "--encoder",
    action="store_true",
    help="The encoder's stages (StagedDeviceEncoder) instead of the decoder's.",
)
parser.add_argument(
    "--first-core",
    type=int,
    default=0,
    help="Without --stage: launch one process per stage on consecutive logical cores "
    "starting here. The cores must be free -- stop the server first.",
)
parser.add_argument(
    "--vae-parallel",
    type=int,
    default=1,
    help="The stage config's vae_parallel: >1 compiles the row-split decoder's stages.",
)
parser.add_argument(
    "--world-size", type=int, default=32, help="With --vae-parallel: the stage's rank count."
)
parser.add_argument("--rank", type=int, default=None, help=argparse.SUPPRESS)
args = parser.parse_args()

STAGES = ["in+mid", "up0", "up1", "up2", "up3", "up4", "out"]
ENCODER_STAGES = ["down0", "down1", "down2", "down3", "down4", "mid+out"]


def _spawn(children) -> int:
    """Run this script once per ``(name, core, extra args, extra env)``; wait for all."""
    procs = []
    for name, core, extra, env in children:
        cmd = [sys.executable, os.path.abspath(__file__), *extra, "--model-path", args.model_path]
        cmd += ["--height", str(args.height), "--width", str(args.width), "--dtype", args.dtype]
        env = dict(os.environ, NEURON_RT_VISIBLE_CORES=str(core), **env)
        procs.append((name, subprocess.Popen(cmd, env=env)))
    failed = [name for name, proc in procs if proc.wait() != 0]
    print(f"FAILED: {failed}" if failed else "all stages compiled", flush=True)
    return 1 if failed else 0


def _launch_all() -> int:
    """One child per stage, each pinned to its own core."""
    stages = ENCODER_STAGES if args.encoder else STAGES
    return _spawn(
        (name, args.first_core + offset, ["--stage", name] + (["--encoder"] if args.encoder else []), {})
        for offset, name in enumerate(stages)
    )


def _launch_sharded() -> int:
    """The stage's world, one process per core; block b compiles stage b (see the top)."""
    if len(STAGES) * args.vae_parallel > args.world_size:
        raise SystemExit(f"{len(STAGES)} stages x {args.vae_parallel} ranks exceed the world")
    world = ["--vae-parallel", str(args.vae_parallel), "--world-size", str(args.world_size)]
    env = {
        "NEURON_RT_ROOT_COMM_ID": "localhost:61234",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": os.environ.get("MASTER_PORT", "29571"),
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "lo"),
    }
    return _spawn((rank, rank, ["--rank", str(rank), *world], env) for rank in range(args.world_size))


if args.vae_parallel > 1 and args.rank is None:
    if args.encoder or args.stage:
        raise SystemExit("--vae-parallel compiles every decoder stage; drop --encoder / --stage")
    sys.exit(_launch_sharded())
if args.stage is None and args.rank is None:
    sys.exit(_launch_all())
_valid = ENCODER_STAGES if args.encoder else STAGES
if args.rank is None and args.stage not in _valid:
    raise SystemExit(f"--stage must be one of {_valid}")

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "600")

import vllm_omni_neuron.bootstrap  # noqa: F401, E402  isort: skip  must precede vllm imports
import torch  # noqa: E402

torch.nn.functional.gelu = torch.ops.aten.gelu.default

from vllm.transformers_utils.config import get_config  # noqa: E402

from vllm_omni_neuron.diffusion.models.hunyuan_image3.vae_device import (  # noqa: E402
    ShardedDeviceDecoder,
    StagedDeviceDecoder,
    StagedDeviceEncoder,
    band,
    bootstrap_device_collectives,
    decoder_stages,
    encoder_stages,
    shard_plan,
    vae_row_groups,
)
from vllm_omni_neuron.lite_compat import (  # noqa: E402
    ensure_current_device_index,
)
from vllm_omni_neuron.lite_compat import initialize as initialize_lite  # noqa: E402


def _stage_input_shapes(vae_config, latent_shape):
    """Each stage's input shape, from one pass over a meta-device decoder."""
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    with torch.device("meta"):
        vae = AutoencoderKLConv3D.from_config({**vae_config, "only_decoder": True})
        hidden = torch.empty(latent_shape)
        shapes = {}
        with torch.no_grad():
            for name, fn in decoder_stages(vae):
                shapes[name] = tuple(hidden.shape)
                hidden = fn(hidden)
    return shapes


def _encoder_stage_input_shapes(vae_config, image_shape):
    """Each encoder stage's input shape, from one pass over a meta-device encoder."""
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    with torch.device("meta"):
        vae = AutoencoderKLConv3D.from_config({**vae_config})
        hidden = torch.empty(image_shape)
        shapes = {}
        with torch.no_grad():
            for name, fn in encoder_stages(vae):
                shapes[name] = tuple(hidden.shape)
                hidden = fn(hidden)
    return shapes


def main_encoder() -> None:
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    config = get_config(args.model_path, trust_remote_code=True)
    vae_config = dict(config.vae)
    image_shape = (1, 3, int(vae_config["ffactor_temporal"]), args.height, args.width)
    shape = _encoder_stage_input_shapes(vae_config, image_shape)[args.stage]

    initialize_lite()
    ensure_current_device_index()
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)
    host_vae = AutoencoderKLConv3D.from_config({**vae_config}).eval()
    encoder = StagedDeviceEncoder(host_vae, vae_config, device, dtype=getattr(torch, args.dtype))
    stage = dict(encoder._stages_for(args.height))[args.stage]
    hidden = torch.zeros(shape, dtype=encoder.dtype).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        stage(hidden).to("cpu")
    print(f"[{args.stage}] input {shape}: compiled and ran in {time.perf_counter() - start:.0f}s")


def main_sharded() -> None:
    """One rank of :func:`_launch_sharded`'s world: the pipeline's groups, then its stage."""
    import torch.distributed as dist
    from vllm_neuron.vllm.platform import NeuronPlatform
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_world_group,
        init_distributed_environment,
    )
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    rank, world, parts = args.rank, args.world_size, args.vae_parallel
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    NeuronPlatform.set_device_count(world)
    initialize_lite()
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    ensure_current_device_index()
    own_group, _ = vae_row_groups(parts)  # the pipeline's _vae_split groups
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)
    bootstrap_device_collectives(device)

    index, band_rank = divmod(rank, parts)
    if index < len(STAGES):
        config = get_config(args.model_path, trust_remote_code=True)
        vae_config = dict(config.vae)
        factor = int(vae_config["ffactor_spatial"])
        latent_height = args.height // factor
        latent_shape = (1, int(vae_config["latent_channels"]), 1, latent_height, args.width // factor)
        host_vae = AutoencoderKLConv3D.from_config({**vae_config, "only_decoder": True}).eval()
        decoder = ShardedDeviceDecoder(
            host_vae, vae_config, device, getattr(torch, args.dtype), band_rank, parts, own_group
        )
        plan = shard_plan(decoder.vae, latent_height, parts)
        name, stage = decoder._stages_for(latent_height)[index]
        # The band's input: the whole latent for in+mid, else the previous stage's band
        # brought to this stage's resolution.
        shape = list(_stage_input_shapes(vae_config, latent_shape)[name])
        if index:
            start, stop = band(plan[index - 1], band_rank, parts)
            shape[-2] = (stop - start) * plan[index]["height"] // plan[index - 1]["height"]
        hidden = torch.zeros(shape, dtype=decoder.dtype).to(device)
        start_time = time.perf_counter()
        with torch.no_grad():
            stage(hidden).to("cpu")
        print(
            f"[{name} rows{band_rank}of{parts}] input {tuple(shape)}: compiled and ran in "
            f"{time.perf_counter() - start_time:.0f}s",
            flush=True,
        )
    # Wait for every block before exiting (gloo; dist.barrier needs an accelerator).
    dist.all_reduce(torch.zeros(1), group=get_world_group().cpu_group)


def main() -> None:
    if args.rank is not None:
        return main_sharded()
    if args.encoder:
        return main_encoder()
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

    config = get_config(args.model_path, trust_remote_code=True)
    vae_config = dict(config.vae)
    factor = int(vae_config["ffactor_spatial"])
    latent_shape = (1, int(vae_config["latent_channels"]), 1)
    latent_shape += (args.height // factor, args.width // factor)
    shape = _stage_input_shapes(vae_config, latent_shape)[args.stage]

    initialize_lite()
    ensure_current_device_index()
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.get_torch_device(0)
    host_vae = AutoencoderKLConv3D.from_config({**vae_config, "only_decoder": True}).eval()
    decoder = StagedDeviceDecoder(host_vae, vae_config, device, dtype=getattr(torch, args.dtype))
    stage = dict(decoder.stages)[args.stage]

    hidden = torch.zeros(shape, dtype=decoder.dtype).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        stage(hidden).to("cpu")
    print(f"[{args.stage}] input {shape}: compiled and ran in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
