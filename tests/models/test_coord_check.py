import torch
import numpy as np
from typing import Callable
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.models.mpt.modeling_mpt_mup import ComposerMPTMuPCausalLM


def _run_coord_step(model: torch.nn.Module, vocab_size: int = 64) -> dict[str, float]:
    hooks = {}
    results: dict[str, list[float]] = {
        'token_embedding': [],
        'attn': [],
        'mlp': [],
        'lm_head': [],
    }

    def _hook_factory(key: str):
        def _hook(_module, _inp, out):
            results[key].append(out.detach().abs().mean().item())
        return _hook

    for name, module in model.named_modules():
        if name == 'transformer.wte':
            hooks[name] = module.register_forward_hook(_hook_factory('token_embedding'))
        elif name.endswith('.attn'):
            hooks[name] = module.register_forward_hook(_hook_factory('attn'))
        elif name.endswith('.mlp'):
            hooks[name] = module.register_forward_hook(_hook_factory('mlp'))
        elif name == 'lm_head':
            hooks[name] = module.register_forward_hook(_hook_factory('lm_head'))

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(2):
        inputs = torch.randint(0, vocab_size, (2, 8))
        labels = torch.randint(0, vocab_size, (2, 8))
        out = model(input_ids=inputs)
        loss = loss_fn(out.logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optim.step()
        optim.zero_grad()

    for handle in hooks.values():
        handle.remove()

    return {k: float(np.mean(v)) for k, v in results.items()}


def _collect_widths(build_fn: Callable[..., ComposerMPTCausalLM], widths: list[int], base_width: int | None = None):
    vals = []
    head_size = 16
    for width in widths:
        n_heads = max(1, width // head_size)
        kwargs = {
            'd_model': width,
            'n_heads': n_heads,
            'attn_config': {'attn_impl': 'torch'},
            'loss_fn': 'torch_crossentropy',
        }
        if base_width is not None:
            kwargs['mup_width_multiplier'] = width / float(base_width)
        model = build_fn(**kwargs)
        vals.append(_run_coord_step(model))
    return vals


def test_coord_check_mup_vs_sp(build_tiny_mpt: Callable[..., ComposerMPTCausalLM],
                               build_tiny_mpt_mup: Callable[..., ComposerMPTMuPCausalLM]):
    torch.manual_seed(0)
    widths = [32, 64]
    sp_vals = _collect_widths(build_tiny_mpt, widths)
    mup_vals = _collect_widths(build_tiny_mpt_mup, widths, base_width=widths[0])

    sp_diff = abs(sp_vals[1]['token_embedding'] - sp_vals[0]['token_embedding'])
    mup_diff = abs(mup_vals[1]['token_embedding'] - mup_vals[0]['token_embedding'])

    assert mup_diff < sp_diff
