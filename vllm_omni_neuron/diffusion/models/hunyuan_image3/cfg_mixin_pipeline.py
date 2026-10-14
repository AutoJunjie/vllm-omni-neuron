# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage3 with the plugin's ``NeuronCFGParallelMixin`` (``model_config.cfg_combine: mixin``).

With ``parallel_config.cfg_parallel_size: 2`` the engine builds two TP replicas and the
pipeline runs one CFG branch on each (prefill and step inputs of that branch). By
default (``cfg_combine: in_graph``) the step graph gathers the two branches' hidden
states and runs the final layer over every rank, so it returns both predictions. With
``cfg_combine: mixin`` the step graph returns this replica's branch only, its final
layer over the replica's TP group, and every step goes through the mixin's
``predict_noise_maybe_with_cfg``: that graph, then the mixin's compiled all-gather +
combine. Text-to-image only, no step cache yet.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def wants_cfg_mixin(model_config: dict | None) -> bool:
    return dict(model_config or {}).get("cfg_combine") == "mixin"


_CLASS = None


def cfg_mixin_pipeline_class(pipeline_cls):
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    from vllm_omni_neuron.diffusion.distributed.cfg_parallel import NeuronCFGParallelMixin

    class NeuronHunyuanImage3CFGMixinPipeline(NeuronCFGParallelMixin, pipeline_cls):
        def __init__(self, od_config, prefix: str = "") -> None:
            super().__init__(od_config, prefix)
            if not self.cfg_parallel:
                raise ValueError("cfg_combine: mixin needs parallel_config.cfg_parallel_size: 2")
            if self._step_cache_config is not None or self._tea_cache_config is not None:
                raise ValueError("cfg_combine: mixin has no step cache / TeaCache yet")
            self._cfg_scale = None
            logger.info(
                "HunyuanImage3: CFG via NeuronCFGParallelMixin, branch %d on TP%d",
                self.model.cp_rank,
                self.model.tp_size,
            )

        def predict_noise(self, call_args):
            fn = self._compiled_denoise_step or self._denoise_step_graph
            return fn(*call_args)

        def _run_denoise_step(self, *call_args):
            combined = self.predict_noise_maybe_with_cfg(
                do_true_cfg=True,
                true_cfg_scale=self._cfg_scale,
                positive_kwargs={"call_args": call_args},
                negative_kwargs={"call_args": call_args},
                cfg_normalize=False,
            )
            # The host loop combines a two-row CFG batch; hand it the combined
            # prediction twice, which its combine returns unchanged (x + g * 0).
            combined = combined.cpu().to(torch.float32)
            return torch.cat((combined, combined))

        def _generate_image(self, model_inputs, *, steps: int, guidance: float):
            if guidance <= 1.0:
                raise ValueError("cfg_combine: mixin needs CFG (guidance_scale > 1)")
            self._cfg_scale = float(guidance)
            return super()._generate_image(model_inputs, steps=steps, guidance=guidance)

    _CLASS = NeuronHunyuanImage3CFGMixinPipeline
    return _CLASS


def install(pipeline_cls) -> None:
    """Make ``pipeline_cls(od_config)`` build the mixin subclass when the config asks."""
    if getattr(pipeline_cls, "_cfg_mixin_installed", False):
        return

    def __new__(cls, *args, **kwargs):
        od_config = kwargs.get("od_config", args[0] if args else None)
        if cls is pipeline_cls and wants_cfg_mixin(getattr(od_config, "model_config", None)):
            cls = cfg_mixin_pipeline_class(pipeline_cls)
        return object.__new__(cls)

    pipeline_cls.__new__ = staticmethod(__new__)
    pipeline_cls._cfg_mixin_installed = True
