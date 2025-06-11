# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""muP-enabled MPT model."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..layers.mup_embedding import MuPSharedEmbedding
from .configuration_mpt_mup import MPTMuPConfig
from .modeling_mpt import ComposerMPTCausalLM, MPTForCausalLM, MPTModel


class MPTMuPModel(MPTModel):
    config_class = MPTMuPConfig

    def __init__(self, config: MPTMuPConfig):
        super().__init__(config)
        if config.mup_enabled:
            # Begin muP code: scale input embeddings
            self.wte = MuPSharedEmbedding(
                config.vocab_size,
                config.d_model,
                padding_idx=config.pad_token_id,
                device=config.init_device,
                scale=config.mup_input_alpha,
            )
            if config.init_device != 'meta':
                self.param_init_fn(self.wte)
            # End muP code
        self.mup_cfg = config
        if self.mup_cfg.mup_enabled:
            # Begin muP code: reinitialize weights following muP rules
            for name, param in self.named_parameters():
                if name.endswith('Wqkv.weight') or name.endswith('fc1_weight'):
                    # NOTE: check if we can get the value from param instead of config
                    nn.init.normal_(
                        param,
                        mean=0.0,
                        std=(
                            init_std if (
                                init_std :=
                                self.mup_cfg.init_config.get('init_std', None)
                            ) is not None else 0.02
                        ) / (self.mup_cfg.mup_width_multiplier**0.5),
                    )
                elif name.endswith(
                    'out_proj.weight',
                ) or name.endswith('fc2_weight'):
                    nn.init.normal_(
                        param,
                        mean=0.0,
                        std=(
                            init_std if (
                                init_std :=
                                self.mup_cfg.init_config.get('init_std', None)
                            ) is not None else 0.02
                        ) / ((
                            2 * self.mup_cfg.n_layers *
                            self.mup_cfg.mup_width_multiplier
                        )**0.5),
                    )
            # End muP code

    def get_optimizer_param_groups(self, weight_decay: float):
        """Return parameter groups with muP-specific LR scaling."""
        param_dict = {
            n: p for n, p in self.named_parameters() if p.requires_grad
        }
        if self.mup_cfg.mup_enabled and not self.mup_cfg.mup_disable_hidden_lr_scaling:
            # Begin muP code: build parameter groups for muP
            mup_decay, decay, nodecay = [], [], []
            for n, p in param_dict.items():
                if p.dim() >= 2:
                    if n.endswith(
                        'Wqkv.weight',
                    ) or n.endswith('fc1_weight') or n.endswith(
                        'out_proj.weight',
                    ) or n.endswith('fc2_weight'):
                        mup_decay.append(p)
                    else:
                        decay.append(p)
                else:
                    nodecay.append(p)
            return [
                {
                    'params': mup_decay,
                    'weight_decay': weight_decay,
                    'lr_scale': 1.0 / self.mup_cfg.mup_width_multiplier,
                },
                {
                    'params': decay,
                    'weight_decay': weight_decay,
                    'lr_scale': 1.0,
                },
                {
                    'params': nodecay,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
            ]
            # End muP code
        else:
            decay = [p for _n, p in param_dict.items() if p.dim() >= 2]
            nodecay = [p for _n, p in param_dict.items() if p.dim() < 2]
            return [
                {
                    'params': decay,
                    'weight_decay': weight_decay,
                },
                {
                    'params': nodecay,
                    'weight_decay': 0.0,
                },
            ]


class MPTMuPForCausalLM(MPTForCausalLM):
    config_class = MPTMuPConfig

    def __init__(self, config: MPTMuPConfig):
        super().__init__(config)
        if not isinstance(self.transformer, MPTMuPModel):
            self.transformer = MPTMuPModel(config)
        self.mup_cfg = config

    def forward(self, *args: Any, **kwargs: Any):
        outputs = super().forward(*args, **kwargs)
        if self.mup_cfg.mup_enabled:
            # Begin muP code: scale output logits
            logits = outputs.logits * (
                self.mup_cfg.mup_output_alpha /
                self.mup_cfg.mup_width_multiplier
            )
            outputs.logits = logits
            # End muP code
        return outputs

    def get_optimizer_param_groups(self, weight_decay: float):
        return self.transformer.get_optimizer_param_groups(weight_decay)


class ComposerMPTCausalLMWithParamGroups(ComposerMPTCausalLM):
    """Composer wrapper that delegates optimizer param group creation to the underlying model."""

    def get_optimizer_param_groups(self, weight_decay: float):
        """Delegate to underlying model to build param groups."""
        return self.model.get_optimizer_param_groups(weight_decay)


class ComposerMPTCausalLMWithParamGroupsMuP(ComposerMPTCausalLMWithParamGroups):

    @property
    def model_class(self) -> type[MPTMuPForCausalLM]:
        return MPTMuPForCausalLM

    @property
    def config_class(self) -> type[MPTMuPConfig]:
        return MPTMuPConfig
