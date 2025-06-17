# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch

from mup_examples.model import GPT, GPTConfig


def _build_model(mup_enabled: bool) -> GPT:
    cfg = GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mup_enabled=mup_enabled,
    )
    return GPT(cfg)


def test_optimizer_groups_mup():
    model = _build_model(True)
    optim = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type='cpu',
    )
    assert len(optim.param_groups) == 3


def test_optimizer_groups_sp():
    model = _build_model(False)
    optim = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type='cpu',
    )
    assert len(optim.param_groups) == 2
