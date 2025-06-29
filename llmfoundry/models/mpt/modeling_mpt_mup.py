# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""muP-enabled MPT model."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..layers.mup_embedding import MuPSharedEmbedding
from .configuration_mpt_mup import MPTMuPConfig
from .modeling_mpt import (
    ComposerMPTCausalLM,
    MPTForCausalLM,
    MPTModel,
)


class MPTMuPModel(MPTModel):
    config_class = MPTMuPConfig

    def __init__(self, config: MPTMuPConfig):
        config._validate_config()
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
                if name.endswith(
                    'Wqkv.weight',
                ) or name.endswith('ffn.up_proj.weight'):
                    # NOTE: check if we can get the value from param instead of config
                    muP_std = (
                        init_std if (
                            init_std :=
                            self.mup_cfg.init_config.get('init_std', None)
                        ) is not None else 0.02
                    ) / (self.mup_cfg.mup_width_multiplier**0.5)
                    nn.init.normal_(
                        param,
                        mean=0.0,
                        std=muP_std,
                    )
                    log.warning(
                        f'Initialized {name} with muP std: {muP_std:.4f}',
                    )

                elif name.endswith(
                    'out_proj.weight',
                ) or name.endswith('ffn.down_proj.weight'):
                    muP_std = (
                        init_std if (
                            init_std :=
                            self.mup_cfg.init_config.get('init_std', None)
                        ) is not None else 0.02
                    ) / ((
                        2 * self.mup_cfg.n_layers *
                        self.mup_cfg.mup_width_multiplier
                    )**0.5)
                    nn.init.normal_(
                        param,
                        mean=0.0,
                        std=muP_std,
                    )
                    log.warning(
                        f'Initialized {name} with muP std: {muP_std:.4f}',
                    )
            # End muP code

    def get_optimizer_param_groups(
        self,
        optimizer_config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return parameter groups with muP-specific LR scaling."""
        weight_decay: float = optimizer_config.pop('weight_decay')
        lr: float = optimizer_config.pop('lr')
        param_dict = {
            n: p for n, p in self.named_parameters() if p.requires_grad
        }
        if self.mup_cfg.mup_enabled and not self.mup_cfg.mup_disable_hidden_lr_scaling:
            # Begin muP code: build parameter groups for muP
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            mup_decay, decay, nodecay = [], [], []
            for n, p in param_dict.items():
                if p.dim() >= 2:
                    if n.endswith(
                        'Wqkv.weight',
                    ) or n.endswith(
                        'out_proj.weight',
                    ) or n.endswith(
                        'ffn.up_proj.weight',
                    ) or n.endswith('ffn.down_proj.weight'):
                        mup_decay.append(p)
                        log.warning(
                            f'Adding {n} to muP decay group with lr: {lr / self.mup_cfg.mup_width_multiplier:.6f}',
                        )
                    else:
                        decay.append(p)
                        log.warning(
                            f'Adding {n} to decay group with lr: {lr:.6f}',
                        )
                else:
                    nodecay.append(p)
                    log.warning(
                        f'Adding {n} to no-decay group with lr: {lr:.6f}',
                    )
            return [
                {
                    'params': mup_decay,
                    'weight_decay': weight_decay,
                    'lr': lr / self.mup_cfg.mup_width_multiplier,
                },
                {
                    'params': decay,
                    'weight_decay': weight_decay,
                    'lr': lr,
                },
                {
                    'params': nodecay,
                    'weight_decay': 0.0,
                    'lr': lr,
                },
            ], optimizer_config
            # End muP code
        else:
            decay = [p for _n, p in param_dict.items() if p.dim() >= 2]
            nodecay = [p for _n, p in param_dict.items() if p.dim() < 2]
            return [
                {
                    'params': decay,
                    'weight_decay': weight_decay,
                    'lr': lr,
                },
                {
                    'params': nodecay,
                    'weight_decay': 0.0,
                    'lr': lr,
                },
            ], optimizer_config

    # Replace forward with method which scales output
    def forward(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast:
        outputs = super().forward(*args, **kwargs)

        if self.mup_cfg.mup_enabled:
            # Begin muP code: scale output logits
            outputs.last_hidden_state = outputs.last_hidden_state * (
                self.mup_cfg.mup_output_alpha /
                self.mup_cfg.mup_width_multiplier
            )
            log.warning(
                f'Scaling output logits by {self.mup_cfg.mup_output_alpha / self.mup_cfg.mup_width_multiplier:.4f}',
            )
            # End muP code

        return outputs


class MPTMuPForCausalLM(MPTForCausalLM):
    config_class = MPTMuPConfig

    def __init__(self, config: MPTMuPConfig):
        super().__init__(config)
        if not isinstance(self.transformer, MPTMuPModel):
            self.transformer = MPTMuPModel(config)

        if config.init_device != 'meta' and self.lm_head is not None:
            self.param_init_fn(self.lm_head)
        self.mup_cfg = config

    def get_optimizer_param_groups(
        self,
        optimizer_config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self.transformer.get_optimizer_param_groups(optimizer_config)


class ComposerMPTCausalLMWithParamGroups(ComposerMPTCausalLM):
    """Composer wrapper that delegates optimizer param group creation to the underlying model."""

    def get_optimizer_param_groups(
        self,
        optimizer_config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Delegate to underlying model to build param groups."""
        return self.model.get_optimizer_param_groups(optimizer_config)


class ComposerMPTCausalLMWithParamGroupsMuP(ComposerMPTCausalLMWithParamGroups):

    @property
    def model_class(self) -> type[MPTMuPForCausalLM]:
        return MPTMuPForCausalLM

    @property
    def config_class(self) -> type[MPTMuPConfig]:
        return MPTMuPConfig
