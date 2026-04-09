"""
Memory manager: CPU ↔ GPU offloading for frozen models.

During PPO, 4 models compete for VRAM (policy, ref, RM, value).
The RM and ref model are only needed at specific steps, so we
offload them to CPU otherwise.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class DeviceManager:
    """Handles temporary GPU placement of frozen models."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    @contextmanager
    def on_device(self, model: nn.Module):
        """Temporarily move a model to GPU, then back to CPU.

        Usage:
            with device_mgr.on_device(ref_model) as ref:
                logprobs = compute_logprobs(ref, ...)
            # ref_model is back on CPU, VRAM freed
        """
        original_device = next(model.parameters()).device
        model.to(self.device)
        try:
            yield model
        finally:
            model.to(original_device)
            if self.device == "cuda":
                torch.cuda.empty_cache()
