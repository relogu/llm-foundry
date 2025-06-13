# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmfoundry.models.layers.mup_embedding import MuPSharedEmbedding
from llmfoundry.models.mpt.configuration_mpt_mup import MPTMuPConfig
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM
from llmfoundry.utils.builders import build_optimizer


def dummy_forward(self, input_ids, *args, **kwargs):
    logits = torch.ones(
        input_ids.shape[0],
        input_ids.shape[1],
        self.config.vocab_size,
    )
    return CausalLMOutputWithPast(logits=logits)


def test_mup_embedding_and_param_groups(build_tiny_mpt_mup):
    model = build_tiny_mpt_mup(mup_input_alpha=1.5)
    assert isinstance(model.model.transformer.wte, MuPSharedEmbedding)
    assert model.model.transformer.wte.scale == 1.5

    groups = model.model.get_optimizer_param_groups(weight_decay=0.1, lr=1.0)
    assert len(groups) == 3
    assert groups[0][
        'lr'] == 1.0 / model.model.transformer.mup_cfg.mup_width_multiplier


def test_mup_logits_scaling(build_tiny_mpt_mup):
    with patch.object(MPTForCausalLM, 'forward', dummy_forward):
        model = build_tiny_mpt_mup(
            mup_width_multiplier=2.0,
            mup_output_alpha=0.5,
        )
        input_ids = torch.ones(2, 4, dtype=torch.long)
        out = model({'input_ids': input_ids})
        expected_scale = 0.5 / 2.0
        assert torch.allclose(
            out.logits,
            torch.ones_like(out.logits) * expected_scale,
        )


def test_mup_softmax_scale():
    cfg = MPTMuPConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=64,
        attn_config={'attn_impl': 'torch'},
        mup_enabled=True,
    )
    head_dim = cfg.d_model // cfg.n_heads
    assert cfg.attn_config['softmax_scale'] == 1.0 / float(head_dim)


def test_build_optimizer_uses_mup_groups(build_tiny_mpt_mup):
    model = build_tiny_mpt_mup(mup_width_multiplier=2.0)
    optim = build_optimizer(
        model.model,
        'decoupled_adamw',
        {
            'lr': 1e-3,
            'weight_decay': 0.01,
        },
    )

    lr_buckets = {p['lr'] for p in optim.param_groups}
    # Assert we have three groups and one is scaled by the width multiplier
    assert len(lr_buckets) == 2
    assert lr_buckets == {1e-3 / 2.0, 1e-3}

    weight_decay_buckets = {p['weight_decay'] for p in optim.param_groups}
    # Assert we have two groups with weight decay and one without
    assert len(weight_decay_buckets) == 2
    assert weight_decay_buckets == {0.01, 0.0}
