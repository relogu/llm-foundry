# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from llmfoundry.callbacks.coord_check import CoordCheckLogger


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Linear(4, 4)
        self.block = torch.nn.Module()
        self.block.attn = torch.nn.Linear(4, 4)
        self.block.ffn = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.wte(x)
        x = self.block.attn(x)
        x = self.block.ffn(x)
        return self.lm_head(x)


def test_coord_check_logger_collects_metrics():
    model = DummyModel()
    cb = CoordCheckLogger()
    logged = []
    state = SimpleNamespace(model=model)
    logger = SimpleNamespace(log_metrics=lambda m: logged.append(m))

    cb.fit_start(state, logger)
    model(torch.randn(2, 4))
    cb.batch_end(state, logger)
    cb.fit_end(state, logger)

    assert logged
    keys = list(logged[0].keys())
    assert 'coord_check/token_embedding' in keys
    assert 'coord_check/attn' in keys
    assert 'coord_check/mlp' in keys
    assert 'coord_check/lm_head' in keys
