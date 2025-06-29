# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch import Tensor

from .custom_embedding import SharedEmbedding


class MuPSharedEmbedding(SharedEmbedding):
    """SharedEmbedding that scales its forward output for muP."""

    def __init__(self, *args: Any, scale: float = 1.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Begin muP code: store scaling factor for embedding outputs
        self.scale = scale
        # End muP code

    def forward(self, input: Tensor, unembed: bool = False) -> Tensor:
        out = super().forward(input, unembed)
        if not unembed:
            # Begin muP code: scale embedding output
            out = out * self.scale
            # End muP code
        return out
