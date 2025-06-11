# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the muP-enabled MPT model."""

from __future__ import annotations

from typing import Any

from .configuration_mpt import MPTConfig


class MPTMuPConfig(MPTConfig):
    """Extends :class:`~llmfoundry.models.mpt.configuration_mpt.MPTConfig` with muP parameters."""

    model_type = 'mpt-mup'

    def __init__(
        self,
        mup_enabled: bool = False,
        mup_disable_attention_scaling: bool = False,
        mup_disable_hidden_lr_scaling: bool = False,
        mup_width_multiplier: float = 1.0,
        mup_input_alpha: float = 1.0,
        mup_output_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Begin muP code: store muP configuration parameters
        self.mup_enabled = mup_enabled
        self.mup_disable_attention_scaling = mup_disable_attention_scaling
        self.mup_disable_hidden_lr_scaling = mup_disable_hidden_lr_scaling
        self.mup_width_multiplier = mup_width_multiplier
        self.mup_input_alpha = mup_input_alpha
        self.mup_output_alpha = mup_output_alpha
        # End muP code

        if self.mup_enabled and not self.mup_disable_attention_scaling:
            head_dim = self.d_model // self.n_heads

            # NOTE: Assumes that d_keys is head_dim
            # softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,

            if self.attn_config.get('softmax_scale') is None:
                self.attn_config = dict(self.attn_config)
                self.attn_config['softmax_scale'] = 1.0 / float(head_dim)

        # End __init__
