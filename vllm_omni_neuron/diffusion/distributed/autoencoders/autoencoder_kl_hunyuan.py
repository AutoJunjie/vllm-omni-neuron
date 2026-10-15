# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0's VAE with vLLM-Omni's tile-parallel decode on Neuron.

vLLM-Omni's ``DistributedAutoencoderKLHunyuan`` splits the latent into the VAE's own
spatial tiles (``tile_sample_min_size`` 384 px, 25% overlap: a 4 x 4 grid at 1024 x
1024), decodes each tile on one of ``vae_patch_parallel_size`` ranks and blends the
seams. This subclass keeps that split and blend and runs them where the Neuron backend
can:

- ``tile_split`` selects the tiles with ``index_select`` on the device latent (a slice
  view would be an uncompiled device op under the Lite runtime), as the Wan2.2 VAE does;
- ``tile_exec`` decodes a tile with the staged device decoder (``vae_device.py``: 16-bit,
  float32 GroupNorm reductions, one compiled graph per stage and tile shape);
- the payload gather and the blend are the plugin's ``_NeuronDistributedVaeExecutor``
  (``autoencoder_kl_wan.py``): compiled pack, streamed device gather, compiled blend --
  the same linear ramps over the overlap as the upstream ``blend_v`` / ``blend_h``.

The tiles are decoded independently, so the seams are blended rather than exact (unlike
the pipeline's row split, ``vae_parallel``, whose GroupNorm statistics are all-reduced
over the whole image). At 1024 x 1024 over 16 ranks: 1.11 s against the row split's
1.16 s on 2, the image 42.0 dB from it (max 36/255, at the seams).
"""

from __future__ import annotations

import logging
import time

import torch
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import (
    DistributedAutoencoderKLHunyuan,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    GridSpec,
    TileTask,
)

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    _NeuronDistributedVaeExecutor,
)

logger = logging.getLogger(__name__)


class NeuronDistributedAutoencoderKLHunyuan(DistributedAutoencoderKLHunyuan):
    """``DistributedAutoencoderKLHunyuan`` whose tiles decode on the Neuron device."""

    def init_distributed(self):
        self.distributed_executor = _NeuronDistributedVaeExecutor()
        self._tile_decoder = None
        self._tile_decoder_factory = None
        self._host_latent = None
        self._tile_device = None

    def set_tile_decoder_factory(self, factory) -> None:
        """``factory()`` -> the staged device decoder, built on this rank's first tile."""
        self._tile_decoder_factory = factory

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        # The tiles come from the host copy of the latent: selecting them on the device
        # is an eager device op per tile per rank under the Lite runtime (~4.5 s of a
        # 5 s decode at 16 tiles); each rank moves only its own tiles over in tile_exec.
        host = self._host_latent if self._host_latent is not None else z
        _, _, _, height, width = host.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = int(self.tile_sample_min_size - blend_extent)
        tiles = []
        for i in range(0, height, overlap_size):
            for j in range(0, width, overlap_size):
                tile = host[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tiles.append(
                    TileTask(
                        len(tiles),
                        (i // overlap_size, j // overlap_size),
                        tile.contiguous(),
                        workload=tile.shape[3] * tile.shape[4],
                    )
                )
        spatial = self.tile_sample_min_size // self.tile_latent_min_size
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiles[-1].grid_coord[0] + 1, tiles[-1].grid_coord[1] + 1),
            tile_spec={
                "blend_extent": blend_extent,
                "row_limit": row_limit,
                "sample_height": height * spatial,
                "sample_width": width * spatial,
            },
            output_dtype=z.dtype,
        )
        return tiles, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        if self._tile_decoder is None:
            self._tile_decoder = self._tile_decoder_factory()
        start = time.perf_counter()
        hidden = task.tensor.to(self._tile_device) if task.tensor.device.type == "cpu" else task.tensor
        with torch.no_grad():
            for _, stage in self._tile_decoder.stages:
                hidden = stage(hidden)
        if task.tensor.shape[2] == 1:
            # A single-frame latent keeps the last decoded frame (AutoencoderKLConv3D.decode).
            hidden = self.distributed_executor._compile_device_graph(
                "hunyuan_last_frame", tuple(hidden.shape), lambda x: x[:, :, -1:].contiguous()
            )(hidden)
        self.distributed_executor._await_device_tensor(hidden)
        logger.debug(
            "HunyuanImage3 VAE tile %s %s decoded in %.3fs",
            task.grid_coord,
            tuple(task.tensor.shape[-2:]),
            time.perf_counter() - start,
        )
        return hidden

    def tile_merge(self, local_tile_tensor, meta_gather, grid_spec: GridSpec, tid_coord_map: dict):
        spec = grid_spec.tile_spec
        start = time.perf_counter()
        merged = self.distributed_executor.gather_and_blend_tiles(
            local_tile_tensor,
            meta_gather,
            grid_spec,
            tid_coord_map,
            full_height=spec["sample_height"],
            full_width=spec["sample_width"],
            stride_height=spec["row_limit"],
            stride_width=spec["row_limit"],
            blend_height=spec["blend_extent"],
            blend_width=spec["blend_extent"],
            clamp=False,
        )
        logger.debug("HunyuanImage3 VAE tiles gathered and blended in %.3fs", time.perf_counter() - start)
        return merged

    def spatial_tiled_decode(self, z: torch.Tensor, host_latent: torch.Tensor | None = None):
        """The tile-parallel decode of a device latent; the image lands on rank 0 only.

        ``host_latent`` is the same latent on the host, which the tiles are cut from.
        """
        if not self.is_distributed_enabled():
            return super().spatial_tiled_decode(z)
        self._host_latent, self._tile_device = host_latent, z.device
        try:
            return self.distributed_executor.execute(
                z,
                DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
                broadcast_result=False,
            )
        finally:
            self._host_latent = None
