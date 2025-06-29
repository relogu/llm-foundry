# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from .blocks import MPTBlock

log = logging.getLogger(__name__)


class MPTCompletePBlock(MPTBlock):
    """MPTBlock that scales residual branches for CompleteP."""

    def __init__(
        self,
        depth_alpha_enabled: bool = False,
        depth_multiplier: float = 1.0,
        depth_alpha_exp: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.residual_scaling = (
            1.0 /
            (depth_multiplier**depth_alpha_exp) if depth_alpha_enabled else 1.0
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        rotary_emb_w_meta_info: Optional[dict] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        output_attentions: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
        prev_layer_key_value: Optional[tuple[torch.Tensor,
                                             torch.Tensor]] = None,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[
        torch.Tensor, torch.Tensor]]]:
        extra_kwargs = {}
        if prev_layer_key_value is not None:
            extra_kwargs['prev_layer_key_value'] = prev_layer_key_value
        if key_value_states is not None:
            extra_kwargs['key_value_states'] = key_value_states

        if self.fuse_norm_attn_norm:
            a = self.norm_attn_norm.norm_1(x)
            b, attn_weights, past_key_value = self.norm_attn_norm.attn(
                a,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=is_causal,
                needs_weights=output_attentions,
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
                **extra_kwargs,
            )
            x = x + self.norm_attn_norm.resid_attn_dropout(
                b,
            ) * self.residual_scaling
            log.warning(
                f'CompletePBlock: residual scaling attention {self.residual_scaling:.4f}',
            )
            m = x
            if self.norm_attn_norm.norm_2 is not None:
                m = self.norm_attn_norm.norm_2(x)
        else:
            a = self.norm_1(x)
            b, attn_weights, past_key_value = self.attn(
                a,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=is_causal,
                needs_weights=output_attentions,
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
                **extra_kwargs,
            )
            x = x + self.resid_attn_dropout(b) * self.residual_scaling
            log.warning(
                f'CompletePBlock: residual scaling attention {self.residual_scaling:.4f}',
            )
            m = x
            if self.norm_2 is not None:
                m = self.norm_2(x)

        n = self.apply_ffn(attention_mask, m)
        x = x.to(device=n.device) + self.resid_ffn_dropout(n).to(
            device=n.device,
        ) * self.residual_scaling
        log.warning(
            f'CompletePBlock: residual scaling ffn {self.residual_scaling:.4f}',
        )
        return x, attn_weights, past_key_value
