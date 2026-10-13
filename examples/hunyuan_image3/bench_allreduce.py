# SPDX-License-Identifier: Apache-2.0
"""What does one TP32 all-reduce of a HunyuanImage-3.0 denoise step cost, and does it
overlap with compute in the same graph?

A warm 1024x1024 step is 0.760 s and its pieces of compute add up to ~0.35 s
(``bench_denoise_layer.py``); the ~0.4 s left over is attributed to the two all-reduces
per layer -- 64 per step, ~6 ms each, of the [2, 4097, 4096] bf16 layer output (67 MB).
This measures them directly, under the same Lite runtime, CCOM settings and worker
initialisation as ``serve.sh``, with the all-reduce inside a compiled graph:

* ``allreduce``: a chain of ``k`` in-place all-reduces over the TP group, for each
  ``k`` in ``--chain`` and each token count in ``--tokens``. The slope over ``k`` is the
  per-collective cost with the graph launch taken out.
* ``overlap``: one graph with two independent branches -- A all-reduces a layer output
  ``--overlap-k`` times, B runs ``--matmuls`` chained [tokens, 4096] x [4096, 4096]
  matmuls -- against A alone and B alone. If the runtime overlaps them, A+B in one
  graph takes about max(A, B), not A + B.

Every timed call is preceded by a host barrier so the ranks start together, and timed
on every rank; the reported time is the slowest rank's.

Measured on trn2.48xlarge, cores 0-31 (2026-10-12): 67 MB takes 1.07 ms (busbw 122
GB/s) and 33.5 MB 0.86 ms, so the 64 all-reduces are ~68 ms of the step, not ~0.4 s.
The two branches do not overlap: A+B in one graph is 4-8 ms *longer* than A and B
run apart, with or without ``--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2``.

    torchrun --nproc-per-node 32 examples/hunyuan_image3/bench_allreduce.py
    torchrun --nproc-per-node 32 examples/hunyuan_image3/bench_allreduce.py --only overlap \\
        --tensorizer-options "--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2"
"""

import argparse
import os
import time

# Per-rank device visibility and CCOM root, as NeuronDiffusionWorker.init_device sets
# them; the rest is serve.sh's environment.
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
os.environ["NEURON_RT_VISIBLE_CORES"] = str(LOCAL_RANK)
os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61234"
os.environ.pop("NEURON_LIBRARY_PATH", None)
os.environ.setdefault("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "2048")
os.environ.setdefault("NEURON_RT_DBG_CC_DMA_PACKET_SIZE", "2048")
os.environ.setdefault("NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE", "167772160")
os.environ.setdefault("NEURON_CC_FLAGS", "-O1 --hbm-scratchpad-page-size=2048")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "300")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import vllm_omni_neuron.bootstrap  # noqa: E402, F401  isort: skip  must precede vllm imports
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

HIDDEN = 4096
MODES = ("allreduce", "overlap", "tpcp")
DEFAULT_MODES = ("allreduce", "overlap")
HOST_GROUP = None

parser = argparse.ArgumentParser(description="TP all-reduce cost inside a compiled graph")
parser.add_argument(
    "--tokens",
    type=int,
    nargs="+",
    default=[2 * 4097, 4097],
    help="Rows of the [tokens, 4096] bf16 buffer: 8194 = the CFG-batched layer output "
    "(67 MB), 4097 = one CFG branch (33.5 MB).",
)
parser.add_argument("--chain", type=int, nargs="+", default=[1, 4, 16])
parser.add_argument("--overlap-tokens", type=int, default=2 * 4097)
parser.add_argument("--overlap-k", type=int, default=8, help="All-reduces in branch A.")
parser.add_argument("--matmuls", type=int, default=4, help="Chained matmuls in branch B.")
parser.add_argument("--runs", type=int, default=10)
parser.add_argument(
    "--tensorizer-options",
    default=None,
    help="Passed as neuronx-cc --tensorizer-options, e.g. NxDI's "
    "'--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'. Unset = the "
    "pipeline's compiler args exactly.",
)
parser.add_argument(
    "--compiler-arg",
    action="append",
    default=[],
    help="Extra compiler argument (repeatable). The native backend hands compiler_args "
    "to its in-process compiler through NEURON_CC_FLAGS, and ignores the variable "
    "itself once compiler_args is set, so flags must go here.",
)
parser.add_argument("--sub-tp", type=int, default=16, help="tpcp: TP group size; CP = world / sub-tp.")
parser.add_argument(
    "--tpcp-tokens",
    type=int,
    nargs="+",
    default=[2049, 4097],
    help="tpcp: rows of the all-reduced activation; 2049 = half the 4097 image tokens (CP2).",
)
parser.add_argument("--kv-tokens", type=int, default=2049, help="tpcp: rows of each rank's K and V.")
parser.add_argument("--only", nargs="*", default=None, choices=MODES)
args = parser.parse_args()


def _init_worker():
    """NeuronDiffusionWorker.init_device's native + Lite path, minus the model."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_neuron import envs as neuron_envs
    from vllm_neuron.vllm.platform import NeuronPlatform
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.platforms import current_omni_platform

    from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups
    from vllm_omni_neuron.lite_compat import initialize as initialize_lite

    world = int(os.environ["WORLD_SIZE"])
    NeuronPlatform.set_device_count(world)
    initialize_lite()
    torch.set_num_threads(1)
    # One visible core per process, so the tensors go to its index 0.
    device = current_omni_platform.get_torch_device(0)
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    _init_worker._context = context
    init_distributed_environment(
        world_size=world,
        rank=int(os.environ["RANK"]),
        local_rank=LOCAL_RANK,
        backend=neuron_envs.get_dist_backend(),
    )
    torch.nn.functional.gelu = torch.ops.aten.gelu.default
    initialize_model_parallel(tensor_parallel_size=world)
    register_replica_groups(tp_size=world, cp_size=1)
    from vllm.distributed.parallel_state import get_tp_group

    # Host-side barriers and timing reductions stay off the device.
    return device, get_tp_group().device_group, dist.new_group(backend="gloo"), world


def _host_barrier():
    # dist.barrier asks for the accelerator even on a gloo group, which Lite does not
    # register; a CPU all-reduce on the gloo group synchronises the ranks the same way.
    dist.all_reduce(torch.zeros(1), group=HOST_GROUP)


def _compile(fn, name):
    compiler_args = [
        "--model-type=transformer",
        "--auto-cast=none",
        "-O1",
        "--hbm-scratchpad-page-size=2048",
    ]
    if args.tensorizer_options:
        compiler_args.append(f"--tensorizer-options={args.tensorizer_options}")
        name = f"{name}_tz"
    compiler_args += args.compiler_arg
    return torch.compile(
        fn,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": f"bench_allreduce_{name}",
            "compiler_args": compiler_args,
        },
    )


def _time(name, fn, *inputs):
    """Slowest rank's per-call time; each call starts behind a host barrier.

    The graph returns a scalar that depends on every branch, so copying it back is the
    sync and the outputs never leave the device.
    """
    compiled = _compile(fn, name)
    start = time.perf_counter()
    compiled(*inputs).to("cpu")
    compile_seconds = time.perf_counter() - start
    compiled(*inputs).to("cpu")
    times = []
    for _ in range(args.runs):
        _host_barrier()
        start = time.perf_counter()
        compiled(*inputs).to("cpu")
        times.append(time.perf_counter() - start)
    # Slowest rank per run: a collective finishes when its last member does.
    per_run = torch.tensor(times, dtype=torch.float64)
    dist.all_reduce(per_run, op=dist.ReduceOp.MAX, group=HOST_GROUP)
    ordered = sorted(per_run.tolist())
    best, median = ordered[0], ordered[len(ordered) // 2]
    if dist.get_rank() == 0:
        print(
            f"[{name}] min {best * 1e3:8.3f} ms  median {median * 1e3:8.3f} ms"
            f"  (compile {compile_seconds:.0f}s)",
            flush=True,
        )
    return best, median


def _slope(results):
    """Least-squares per-collective time over chain lengths, and the fixed part."""
    ks = sorted(results)
    mean_k = sum(ks) / len(ks)
    mean_t = sum(results[k] for k in ks) / len(ks)
    slope = sum((k - mean_k) * (results[k] - mean_t) for k in ks) / sum((k - mean_k) ** 2 for k in ks)
    return slope, mean_t - slope * mean_k


def main() -> None:
    global HOST_GROUP
    device, group, HOST_GROUP, world = _init_worker()
    rank = dist.get_rank()
    bf16 = torch.bfloat16
    torch.manual_seed(rank)

    def dev(*shape, scale=1.0):
        return (torch.randn(*shape, dtype=torch.float32) * scale).to(bf16).to(device)

    if rank == 0:
        print(f"TP{world} all-reduce over {group}; device {device}", flush=True)
    want = set(args.only or DEFAULT_MODES)
    bias = dev(1, HIDDEN)

    def ar_chain(k):
        def fn(x, b):
            # An intermediate, as the layer output is, rather than a graph input.
            y = x + b
            for _ in range(k):
                dist.all_reduce(y, group=group)
            # The whole buffer feeds the result, so nothing can narrow the collective.
            return y.float().sum()

        return fn

    if "allreduce" in want:
        # The compiled collective must be a real sum over the group: check it once
        # against the host before trusting any timing.
        probe = dev(args.tokens[0], HIDDEN)

        def reduce_once(x):
            y = x * 1.0
            dist.all_reduce(y, group=group)
            return y

        got = _compile(reduce_once, "check")(probe).to("cpu").float()
        expected = probe.to("cpu").float()
        dist.all_reduce(expected, group=HOST_GROUP)
        error = ((got - expected).norm() / expected.norm()).item()
        if rank == 0:
            print(f"[check] compiled all-reduce vs host sum: relative L2 {error:.2e}", flush=True)
        del probe
        for tokens in args.tokens:
            x = dev(tokens, HIDDEN, scale=0.0)
            megabytes = tokens * HIDDEN * 2 / 1e6
            results = {
                k: _time(f"ar_{tokens}x{k}", ar_chain(k), x, bias)[0] for k in args.chain
            }
            ks = sorted(results)
            if rank == 0 and len(ks) > 1:
                # Least-squares slope over k: per-collective time without the launch.
                mean_k = sum(ks) / len(ks)
                mean_t = sum(results[k] for k in ks) / len(ks)
                slope = sum((k - mean_k) * (results[k] - mean_t) for k in ks) / sum(
                    (k - mean_k) ** 2 for k in ks
                )
                algbw = megabytes / 1e3 / slope
                busbw = algbw * 2 * (world - 1) / world
                print(
                    f"[allreduce] {tokens} x {HIDDEN} bf16 = {megabytes:.1f} MB: "
                    f"{slope * 1e3:.3f} ms per all-reduce (fixed {(mean_t - slope * mean_k) * 1e3:.3f} ms)"
                    f"  algbw {algbw:.1f} GB/s  busbw {busbw:.1f} GB/s"
                    f"  -> x64 = {slope * 64:.3f} s/step",
                    flush=True,
                )
            del x

    if "overlap" in want:
        tokens, k, m = args.overlap_tokens, args.overlap_k, args.matmuls
        x = dev(tokens, HIDDEN, scale=0.0)
        h = dev(tokens, HIDDEN)
        w = dev(HIDDEN, HIDDEN, scale=HIDDEN**-0.5)

        def comm(x, b):
            y = x + b
            for _ in range(k):
                dist.all_reduce(y, group=group)
            return y

        def compute(h, w):
            for _ in range(m):
                h = torch.matmul(h, w)
            return h

        def only_a(x, b, h, w):
            return comm(x, b).float().sum()

        def only_b(x, b, h, w):
            return compute(h, w).float().sum()

        def both(x, b, h, w):
            return comm(x, b).float().sum() + compute(h, w).float().sum()

        a = _time(f"overlap_a_{tokens}_k{k}", only_a, x, bias, h, w)[0]
        b = _time(f"overlap_b_{tokens}_m{m}", only_b, x, bias, h, w)[0]
        ab = _time(f"overlap_ab_{tokens}_k{k}_m{m}", both, x, bias, h, w)[0]
        if rank == 0:
            hidden_fraction = (a + b - ab) / min(a, b)
            print(
                f"[overlap] A (all-reduce x{k}) {a * 1e3:.2f} ms, B (matmul x{m}) "
                f"{b * 1e3:.2f} ms, A+B in one graph {ab * 1e3:.2f} ms; "
                f"sum {(a + b) * 1e3:.2f}, max {max(a, b) * 1e3:.2f}; "
                f"{hidden_fraction:.0%} of the shorter branch hidden",
                flush=True,
            )


    if "tpcp" in want:
        # TP x CP on the same ranks: contiguous TP blocks (block b = chips 4b..4b+3, one
        # torus row at TP16), CP across the blocks at the same position (vertically
        # adjacent chips). Every rank builds every group, then registers the partition
        # under its own group's name so the compiled collectives legalize.
        from vllm_omni_neuron.lite_compat import register_process_group_replica_groups

        sub = args.sub_tp
        tp_parts = [list(range(b, b + sub)) for b in range(0, world, sub)]
        cp_parts = [list(range(i, world, sub)) for i in range(sub)]
        tp_pg = cp_pg = None
        for part in tp_parts:
            pg = dist.new_group(part)
            tp_pg = pg if rank in part else tp_pg
        for part in cp_parts:
            pg = dist.new_group(part)
            cp_pg = pg if rank in part else cp_pg
        register_process_group_replica_groups(tp_pg.group_name, tp_parts)
        register_process_group_replica_groups(cp_pg.group_name, cp_parts)
        cp = len(cp_parts[0])
        if rank == 0:
            print(f"[tpcp] TP{sub} groups {[p[0] for p in tp_parts]}+.., CP{cp} groups e.g. {cp_parts[0]}", flush=True)

        def tp_chain(k):
            def fn(x, b):
                y = x + b
                for _ in range(k):
                    dist.all_reduce(y, group=tp_pg)
                return y.float().sum()

            return fn

        for tokens in args.tpcp_tokens:
            x = dev(tokens, HIDDEN, scale=0.0)
            megabytes = tokens * HIDDEN * 2 / 1e6
            results = {k: _time(f"tp{sub}_ar_{tokens}x{k}", tp_chain(k), x, bias)[0] for k in args.chain}
            if rank == 0:
                slope, fixed = _slope(results)
                busbw = megabytes / 1e3 / slope * 2 * (sub - 1) / sub
                print(
                    f"[tpcp] TP{sub} all-reduce {tokens} x {HIDDEN} = {megabytes:.1f} MB: "
                    f"{slope * 1e3:.3f} ms (fixed {fixed * 1e3:.3f} ms), busbw {busbw:.1f} GB/s, x64 = {slope * 64:.3f} s",
                    flush=True,
                )
            del x

        def cp_chain(k):
            def fn(kv):
                y = kv
                for _ in range(k):
                    out = y.new_empty((cp * y.shape[0], *y.shape[1:]))
                    dist.all_gather_into_tensor(out, y.contiguous(), group=cp_pg)
                    # Keep the shape and depend on every rank's slice.
                    y = out.reshape(cp, *y.shape).sum(0) * (1.0 / cp)
                return y.float().sum()

            return fn

        kv = dev(2, args.kv_tokens, 128)  # one KV head's K and V for this rank's tokens
        kv_mb = kv.numel() * 2 / 1e6
        results = {k: _time(f"cp{cp}_ag_kv{args.kv_tokens}x{k}", cp_chain(k), kv)[0] for k in args.chain}
        if rank == 0:
            slope, fixed = _slope(results)
            print(
                f"[tpcp] CP{cp} all-gather of K+V {kv_mb:.2f} MB per rank: {slope * 1e3:.3f} ms "
                f"(fixed {fixed * 1e3:.3f} ms), x32 layers = {slope * 32 * 1e3:.1f} ms",
                flush=True,
            )

    _host_barrier()


if __name__ == "__main__":
    main()
