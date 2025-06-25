# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Wrapper using :class:`MPTMuPForCausalLM` for the muP examples."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from llmfoundry.models.mpt.configuration_mpt_mup import MPTMuPConfig
from llmfoundry.models.mpt.modeling_mpt_mup import MPTMuPForCausalLM
from llmfoundry.utils.builders import build_optimizer


class GPTConfig(MPTMuPConfig):
    """Configuration matching the old ``GPTConfig`` interface from nanogpt."""

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        init_std: float = 0.02,
        mup_enabled: bool = False,
        mup_disable_attention_scaling: bool = False,
        mup_disable_hidden_lr_scaling: bool = False,
        mup_width_multiplier: float = 1.0,
        mup_input_alpha: float = 1.0,
        mup_output_alpha: float = 1.0,
        force_weight_tying: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            d_model=n_embd,
            n_heads=n_head,
            n_layers=n_layer,
            max_seq_len=block_size,
            vocab_size=vocab_size,
            resid_pdrop=dropout,
            emb_pdrop=dropout,
            no_bias=not bias,
            init_config={
                'name': 'baseline_',
                'init_std': init_std,
                'emb_init_std': init_std,
            },
            mup_enabled=mup_enabled,
            mup_disable_attention_scaling=mup_disable_attention_scaling,
            mup_disable_hidden_lr_scaling=mup_disable_hidden_lr_scaling,
            mup_width_multiplier=mup_width_multiplier,
            mup_input_alpha=mup_input_alpha,
            mup_output_alpha=mup_output_alpha,
            attn_config={
                'attn_impl': 'torch',
            },
            loss_fn='torch_crossentropy',
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Store arguments for backwards compatibility
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.init_std = init_std
        self.force_weight_tying = force_weight_tying


class GPT(MPTMuPForCausalLM):
    """Thin wrapper around :class:`MPTMuPForCausalLM` with helper utilities."""

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        if config.force_weight_tying:
            assert self.lm_head is not None, 'lm_head must be defined for weight tying'
            self.lm_head.weight = self.transformer.wte.weight

    # ----------------------------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------------------------
    def configure_optimizers(
        self,
        optimizer_name: str,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Return optimizer configured using :func:`build_optimizer`."""
        return build_optimizer(
            self,
            name=optimizer_name,
            optimizer_config={
                'lr': learning_rate,
                'betas': betas,
                'weight_decay': weight_decay,
            },
        )

    # ----------------------------------------------------------------------------
    # Utility methods retained from the original implementation
    # ----------------------------------------------------------------------------
    def crop_block_size(self, block_size: int) -> None:
        """Crop positional embeddings to ``block_size``."""
        assert block_size <= self.config.max_seq_len
        self.config.max_seq_len = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(
                self.transformer.wpe.weight[:block_size],
            )

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate the model flops utilization."""
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H = cfg.n_layers, cfg.n_heads
        Q = cfg.d_model // cfg.n_heads
        T = cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bfloat16 peak FLOPs
        mfu = flops_achieved / flops_promised
        return mfu
