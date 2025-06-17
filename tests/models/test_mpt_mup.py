# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0


import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmfoundry.models.layers.mup_embedding import MuPSharedEmbedding
from llmfoundry.models.mpt.configuration_mpt_mup import MPTMuPConfig
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM
from mup_examples.model import GPT, GPTConfig
from llmfoundry.utils.builders import build_optimizer


def dummy_forward(
    self: MPTForCausalLM,
    input_ids: torch.Tensor,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> CausalLMOutputWithPast:
    logits = torch.ones(
        input_ids.shape[0],
        input_ids.shape[1],
        self.config.vocab_size,
    )
    return CausalLMOutputWithPast(logits=logits)


def test_mup_embedding_and_param_groups() -> None:
    cfg = GPTConfig(
        n_embd=32,
        n_head=4,
        n_layer=2,
        vocab_size=64,
        mup_enabled=True,
        mup_input_alpha=1.5,
        mup_width_multiplier=2.0,
        attn_config={"attn_impl": "torch"},
    )
    model = GPT(cfg)
    assert isinstance(model.transformer.wte, MuPSharedEmbedding)
    assert model.transformer.wte.scale == 1.5

    groups = model.get_optimizer_param_groups(weight_decay=0.1, lr=1.0)
    assert len(groups) == 3
    assert groups[0]["lr"] == 1.0 / cfg.mup_width_multiplier


def test_mup_softmax_scale() -> None:
    cfg = MPTMuPConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=64,
        attn_config={"attn_impl": "torch"},
        mup_enabled=True,
    )
    head_dim = cfg.d_model // cfg.n_heads
    assert cfg.attn_config["softmax_scale"] == 1.0 / float(head_dim)


def test_build_optimizer_uses_mup_groups() -> None:
    cfg = GPTConfig(
        n_embd=32,
        n_head=4,
        n_layer=2,
        vocab_size=64,
        mup_enabled=True,
        mup_width_multiplier=2.0,
        attn_config={"attn_impl": "torch"},
    )
    model = GPT(cfg)
    optim = build_optimizer(
        model,
        "decoupled_adamw",
        {
            "lr": 1e-3,
            "weight_decay": 0.01,
        },
    )

    lr_buckets = {p["lr"] for p in optim.param_groups}
    # Assert we have three groups and one is scaled by the width multiplier
    assert len(lr_buckets) == 2
    assert lr_buckets == {1e-3 / 2.0, 1e-3}

    weight_decay_buckets = {p["weight_decay"] for p in optim.param_groups}
    # Assert we have two groups with weight decay and one without
    assert len(weight_decay_buckets) == 2
    assert weight_decay_buckets == {0.01, 0.0}
