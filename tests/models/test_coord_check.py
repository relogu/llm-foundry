# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from typing import Callable

import numpy as np
import torch

from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.models.mpt.modeling_mpt_mup import \
    ComposerMPTCausalLMWithParamGroupsMuP


def _run_coord_step(model: torch.nn.Module,
                    vocab_size: int = 50368) -> dict[str, float]:
    hooks = {}
    results: dict[str, list[float]] = {
        'token_embedding': [],
        'attn': [],
        'mlp': [],
        'lm_head': [],
    }

    def _hook_factory(key: str):

        def _hook(_module, _inp, out):
            results[key].append(out[0].detach().abs().mean().item())

        return _hook

    for name, module in model.named_modules():
        if 'wte' in name:
            hooks[name] = module.register_forward_hook(
                _hook_factory('token_embedding'),
            )
        elif '.attn' in name:
            hooks[name] = module.register_forward_hook(_hook_factory('attn'))
        elif 'ffn' in name:
            hooks[name] = module.register_forward_hook(_hook_factory('mlp'))

    optim = torch.optim.AdamW(model.parameters(), lr=1e-1)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(2):
        inputs = torch.randint(0, vocab_size, (2, 8))
        labels = torch.randint(0, vocab_size, (2, 8))

        out = model({'input_ids': inputs})

        loss = loss_fn(out.logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optim.step()
        optim.zero_grad()

    for handle in hooks.values():
        handle.remove()

    return {k: float(np.mean(v)) for k, v in results.items()}


def _collect_widths(
    build_fn: Callable[..., ComposerMPTCausalLM],
    widths: list[int],
    base_width: int | None = None,
):
    vals = []
    head_size = 16
    for width in widths:
        n_heads = max(1, width // head_size)
        kwargs = {
            'd_model': width,
            'n_heads': n_heads,
            'attn_config': {
                'attn_impl': 'torch',
            },
            'loss_fn': 'torch_crossentropy',
        }
        if base_width is not None:
            kwargs['mup_width_multiplier'] = width / float(base_width)
        model = build_fn(**kwargs)
        vals.append(_run_coord_step(model))
    return vals


def test_coord_check_mup_vs_sp(
    build_tiny_mpt: Callable[..., ComposerMPTCausalLM],
    build_tiny_mpt_mup: Callable[..., ComposerMPTCausalLMWithParamGroupsMuP],
):
    # TODO: Make sure this test is reasonable
    torch.manual_seed(0)
    widths = [16, 32, 64, 128]
    sp_vals = _collect_widths(build_tiny_mpt, widths)
    mup_vals = _collect_widths(build_tiny_mpt_mup, widths, base_width=widths[0])

    # sp_diff = abs(sp_vals[-1]['mlp'] - sp_vals[0]['mlp'])
    # mup_diff = abs(
    #     mup_vals[-1]['mlp'] - mup_vals[0]['mlp'],
    # )
    print('Sp vals:', sp_vals)
    print('MuP vals:', mup_vals)

    raise ValueError(
        'This test is not yet implemented. '
        'Please implement the coordinate check for MuP vs SP models.',
    )
