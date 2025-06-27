# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch

from llmfoundry.models.layers.completep_block import MPTCompletePBlock
from llmfoundry.models.mpt.modeling_mpt_completep import (
    ComposerMPTCausalLMWithParamGroupsCompleteP,
)


def test_completep_residual_scaling(
    build_tiny_mpt_completep: Callable[
        ..., ComposerMPTCausalLMWithParamGroupsCompleteP],
) -> None:
    model = build_tiny_mpt_completep(depth_multiplier=4.0, depth_alpha_exp=0.5)
    block = model.model.transformer.blocks[0]
    assert isinstance(block, MPTCompletePBlock)
    expected = 1.0 / (4.0**0.5)
    assert block.residual_scaling == expected


def test_completep_param_groups(
    build_tiny_mpt_completep: Callable[
        ..., ComposerMPTCausalLMWithParamGroupsCompleteP],
) -> None:
    model = build_tiny_mpt_completep(mup_width_multiplier=2.0)
    groups = model.get_optimizer_param_groups(weight_decay=0.1, lr=1.0)
    assert len(groups) == 5
    # hidden weight params are group 2 with scaled weight decay
    assert groups[2]['weight_decay'] == 0.1 / (1 / 2.0)
    assert torch.isclose(
        torch.tensor(groups[2]['lr']),
        torch.tensor(1.0 * (1 / 2.0)),
    )
