# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pickle
import runpy
from pathlib import Path

import requests


def test_shakespeare_char_prepare(tmp_path, monkeypatch):
    text = 'To be or not to be'

    class Resp:

        def __init__(self, t):
            self.text = t

    monkeypatch.setattr(requests, 'get', lambda url: Resp(text))

    script = Path('scripts/data_prep/shakespeare_char/prepare.py')
    tmp_script = tmp_path / 'prepare.py'
    tmp_script.write_text(script.read_text())

    runpy.run_path(tmp_script)

    with open(tmp_path / 'meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    assert meta['vocab_size'] == len(set(text))
    assert (tmp_path / 'train.bin').exists()
    assert (tmp_path / 'val.bin').exists()
