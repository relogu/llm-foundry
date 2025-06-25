# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

import numpy as np
import torch
from composer.core import Callback, State
from composer.loggers import Logger
from torch.utils.hooks import RemovableHandle

__all__ = ['CoordCheckLogger']


class CoordCheckLogger(Callback):
    """Log layer activation means for muP coordinate checks."""

    def __init__(self) -> None:
        self.handles: list[RemovableHandle] = []
        self.stats: dict[str, list[float]] = {
            'token_embedding': [],
            'attn': [],
            'mlp': [],
            'lm_head': [],
        }

    def reset(self) -> None:
        self.stats: dict[str, list[float]] = {
            'token_embedding': [],
            'attn': [],
            'mlp': [],
            'lm_head': [],
        }

    def _hook_factory(
        self,
        key: str,
    ) -> Callable[[torch.nn.Module, tuple[Any, ...], torch.Tensor], None]:

        def hook(
            _module: torch.nn.Module,
            _inp: tuple[Any, ...],
            out: torch.Tensor,
        ) -> None:
            with torch.no_grad():
                tensor = out[0] if isinstance(out, tuple) else out
                if isinstance(tensor, torch.Tensor):
                    self.stats[key].append(tensor.detach().abs().mean().item())

        return hook

    def _register_hooks(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if 'wte' in name:
                self.handles.append(
                    module.register_forward_hook(
                        self._hook_factory('token_embedding'),
                    ),
                )
            elif '.attn' in name:
                self.handles.append(
                    module.register_forward_hook(self._hook_factory('attn')),
                )
            elif 'ffn' in name or '.mlp' in name:
                self.handles.append(
                    module.register_forward_hook(self._hook_factory('mlp')),
                )
            elif name == 'lm_head':
                self.handles.append(
                    module.register_forward_hook(self._hook_factory('lm_head')),
                )

    def _remove_hooks(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def fit_start(self, state: State, logger: Logger) -> None:
        self._register_hooks(state.model)

    def batch_end(self, state: State, logger: Logger) -> None:
        metrics = {
            f'coord_check/{k}': float(np.mean(v))
            for k, v in self.stats.items()
            if v
        }
        if metrics:
            logger.log_metrics(metrics)
        self.reset()

    def fit_end(self, state: State, logger: Logger) -> None:
        self._remove_hooks()
