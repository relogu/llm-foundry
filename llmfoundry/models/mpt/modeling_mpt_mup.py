# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""muP-enabled MPT model."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from ..layers.mup_embedding import MuPSharedEmbedding
from .configuration_mpt_mup import MPTMuPConfig
from .modeling_mpt import (
    CROSS_ENTROPY_IGNORE_INDEX,
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
                    log.debug(f'Initialized {name} with muP std: {muP_std:.4f}')

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
                    log.debug(f'Initialized {name} with muP std: {muP_std:.4f}')
            # End muP code

    def get_optimizer_param_groups(self, weight_decay: float, lr: float):
        """Return parameter groups with muP-specific LR scaling."""
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
                        log.debug(
                            f'Adding {n} to muP decay group with lr: {lr / self.mup_cfg.mup_width_multiplier:.6f}',
                        )
                    else:
                        decay.append(p)
                        log.debug(
                            f'Adding {n} to decay group with lr: {lr:.6f}',
                        )
                else:
                    nodecay.append(p)
                    log.debug(
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        logits = outputs.last_hidden_state

        if self.mup_cfg.mup_enabled:
            # Begin muP code: scale output logits
            logits = logits * (
                self.mup_cfg.mup_output_alpha /
                self.mup_cfg.mup_width_multiplier
            )
            # End muP code

        if self.lm_head is not None:
            logits = self.lm_head(logits)
        else:
            # move outputs to same device as weights for token embedding
            # needed to support HF `device_map`
            out = logits
            out = out.to(self.transformer.wte.weight.device)
            logits = self.transformer.wte(out, True)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.',
                )
            logits *= self.logit_scale

        # TODO: Decide what to do with softcapping
        # when using muP.
        if self.final_logit_softcapping is not None:
            logits = self.final_logit_softcapping * torch.tanh(
                logits / self.final_logit_softcapping,
            )

        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_optimizer_param_groups(self, weight_decay: float, lr: float):
        return self.transformer.get_optimizer_param_groups(weight_decay, lr)


class ComposerMPTCausalLMWithParamGroups(ComposerMPTCausalLM):
    """Composer wrapper that delegates optimizer param group creation to the underlying model."""

    def get_optimizer_param_groups(self, weight_decay: float, lr: float):
        """Delegate to underlying model to build param groups."""
        return self.model.get_optimizer_param_groups(weight_decay, lr)


class ComposerMPTCausalLMWithParamGroupsMuP(ComposerMPTCausalLMWithParamGroups):

    @property
    def model_class(self) -> type[MPTMuPForCausalLM]:
        return MPTMuPForCausalLM

    @property
    def config_class(self) -> type[MPTMuPConfig]:
        return MPTMuPConfig
