# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""muP-enabled MPT model."""

from __future__ import annotations

from typing import Any, Optional

from composer.metrics import (
    InContextLearningCodeEvalAccuracy,
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
)
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer.models import HuggingFaceModel
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch import nn
from transformers import PreTrainedTokenizerBase

from ..layers.mup_embedding import MuPSharedEmbedding
from .configuration_mpt_mup import MPTMuPConfig
from .modeling_mpt import MPTForCausalLM, MPTModel


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
                    nn.init.normal_(
                        param,
                        mean=0.0,
                        std=self.mup_cfg.init_config.get('init_std', 0.02) /
                        (self.mup_cfg.mup_width_multiplier**0.5),
                    )
                elif name.endswith(
                    'out_proj.weight',
                ) or name.endswith('fc2_weight'):
                    nn.init.normal_(
                        param,
                        mean=0.0,
                        std=self.mup_cfg.init_config.get('init_std', 0.02) / ((
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
            decay = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
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


class ComposerMPTMuPCausalLM(HuggingFaceModel):

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        resolved = om.to_container(om_model_config, resolve=True)
        hf_config = MPTMuPConfig.from_dict(resolved)
        model = MPTMuPForCausalLM(hf_config)

        use_train_metrics = om_model_config.get('use_train_metrics', True)
        train_metrics = [LanguageCrossEntropy(),
                         LanguagePerplexity()] if use_train_metrics else []
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningCodeEvalAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
            allow_embedding_resizing=True,
        )

        self.n_active_params = sum(p.numel() for p in self.parameters())

        loss_fn_config = om_model_config.get('loss_fn', 'fused_crossentropy')
        if loss_fn_config == 'fused_crossentropy':
            try:
                from flash_attn.losses.cross_entropy import \
                    CrossEntropyLoss as FusedCrossEntropyLoss

                self.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
            except Exception:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    'and `pip install .[gpu]` if installing from source or '
                    '`pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.',
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={loss_fn_config} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].',
            )

    def get_optimizer_param_groups(self, weight_decay: float):
        """Delegate to underlying model to build param groups."""
        return self.model.get_optimizer_param_groups(weight_decay)
