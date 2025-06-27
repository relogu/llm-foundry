# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .configuration_mpt_mup import MPTMuPConfig


class MPTCompletePConfig(MPTMuPConfig):
    """Extends :class:`MPTMuPConfig` with CompleteP parameters."""

    model_type = 'mpt-completep'

    def __init__(
        self,
        depth_alpha_enabled: bool = False,
        depth_multiplier: float = 1.0,
        depth_alpha_exp: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.depth_alpha_enabled = depth_alpha_enabled
        self.depth_multiplier = depth_multiplier
        self.depth_alpha_exp = depth_alpha_exp
