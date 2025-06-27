# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from ..layers.completep_block import MPTCompletePBlock
from .configuration_mpt_completep import MPTCompletePConfig
from .modeling_mpt_mup import (
    ComposerMPTCausalLMWithParamGroups,
    MPTMuPForCausalLM,
    MPTMuPModel,
)

log = logging.getLogger(__name__)


class MPTCompletePModel(MPTMuPModel):
    config_class = MPTCompletePConfig

    @property
    def block_class(self) -> type[MPTCompletePBlock]:
        return MPTCompletePBlock

    def get_optimizer_param_groups(
        self,
        optimizer_config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        weight_decay: float = optimizer_config.pop('weight_decay')
        lr: float = optimizer_config.pop('lr')
        param_dict = {
            n: p for n, p in self.named_parameters() if p.requires_grad
        }
        if self.mup_cfg.mup_enabled and not self.mup_cfg.mup_disable_hidden_lr_scaling:
            emb_params = []
            hidden_ln_params = []
            hidden_weight_params = []
            hidden_bias_params = []
            final_ln_params = []
            for n, p in param_dict.items():
                if n.endswith('wte.weight') or n.endswith('wpe.weight'):
                    emb_params.append(p)
                    log.warning(
                        f'Adding {n} to embedding group with lr: {lr:.6f}',
                    )
                elif '.norm_' in n and not '.norm_f.' in n:
                    hidden_ln_params.append(p)
                    log.warning(
                        f'Adding {n} to hidden layer norm group with lr: {lr * self.mup_cfg.depth_multiplier:.6f}',
                    )
                elif n.endswith(
                    'Wqkv.weight',
                ) or n.endswith('out_proj.weight') or n.endswith(
                    'ffn.up_proj.weight',
                ) or n.endswith('ffn.down_proj.weight'):
                    hidden_weight_params.append(p)
                    log.warning(
                        f'Adding {n} to hidden weight group with lr: {lr * self.mup_cfg.mup_width_multiplier * self.mup_cfg.depth_multiplier:.6f}',
                    )
                elif n.endswith(
                    'Wqkv.bias',
                ) or n.endswith('out_proj.bias') or n.endswith(
                    'ffn.up_proj.bias',
                ) or n.endswith('ffn.down_proj.bias'):
                    hidden_bias_params.append(p)
                    log.warning(
                        f'Adding {n} to hidden bias group with lr: {lr * self.mup_cfg.depth_multiplier:.6f}',
                    )
                elif 'norm_f' in n:
                    final_ln_params.append(p)
                    log.warning(
                        f'Adding {n} to final layer norm group with lr: {lr:.6f}',
                    )
                else:
                    log.warning(f'Unhandled parameter {n}')
            width_lr_scaling = 1 / self.mup_cfg.mup_width_multiplier
            depth_lr_scaling = self.mup_cfg.depth_multiplier**(
                self.mup_cfg.depth_alpha_exp - 1
            ) if self.mup_cfg.depth_alpha_enabled else 1.0
            og_eps = optimizer_config['eps']
            optimizer_config['eps'] *= (1 /
                                        self.config.mup_width_multiplier) * (
                                            self.config.depth_multiplier**
                                            (-1 * self.config.depth_alpha_exp)
                                        )
            log.warning(
                f"Using width_lr_scaling: {width_lr_scaling}, depth_lr_scaling: {depth_lr_scaling}, eps: {optimizer_config['eps']} scaled from {og_eps}",
            )
            return [
                {
                    'params': emb_params,
                    'weight_decay': weight_decay,
                    'lr': lr,
                },
                {
                    'params': hidden_ln_params,
                    'weight_decay': 0.0,
                    'lr': lr * depth_lr_scaling,
                },
                {
                    'params': hidden_weight_params,
                    'weight_decay': weight_decay / width_lr_scaling,
                    'lr': lr * width_lr_scaling * depth_lr_scaling,
                },
                {
                    'params': hidden_bias_params,
                    'weight_decay': 0.0,
                    'lr': lr * depth_lr_scaling,
                },
                {
                    'params': final_ln_params,
                    'weight_decay': 0.0,
                    'lr': lr,
                },
            ], optimizer_config
        else:
            decay_params = [p for _n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for _n, p in param_dict.items() if p.dim() < 2]
            return [
                {
                    'params': decay_params,
                    'weight_decay': weight_decay,
                    'lr': lr,
                },
                {
                    'params': nodecay_params,
                    'weight_decay': 0.0,
                    'lr': lr,
                },
            ], optimizer_config


class MPTCompletePForCausalLM(MPTMuPForCausalLM):
    config_class = MPTCompletePConfig

    def __init__(self, config: MPTCompletePConfig):
        super().__init__(config)
        if not isinstance(self.transformer, MPTCompletePModel):
            self.transformer = MPTCompletePModel(config)
        if config.init_device != 'meta' and self.lm_head is not None:
            self.param_init_fn(self.lm_head)
        self.mup_cfg = config

    def get_optimizer_param_groups(
        self,
        optimizer_config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self.transformer.get_optimizer_param_groups(optimizer_config)


class ComposerMPTCausalLMWithParamGroupsCompleteP(
    ComposerMPTCausalLMWithParamGroups,
):

    @property
    def model_class(self) -> type[MPTCompletePForCausalLM]:
        return MPTCompletePForCausalLM

    @property
    def config_class(self) -> type[MPTCompletePConfig]:
        return MPTCompletePConfig
