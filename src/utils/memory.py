from contextlib import contextmanager

import torch


class MemoryManager:
    """Small utility for dynamic CPU/GPU model placement."""

    def __init__(self, gpu_device: str = "cuda"):
        self.gpu_device = gpu_device

    @staticmethod
    def current_allocated_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024**2)

    def move_model(self, model, device: str):
        model.to(device)
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def offload_to_cpu(self, model):
        return self.move_model(model, "cpu")

    @contextmanager
    def temporarily_on_gpu(self, model):
        if not torch.cuda.is_available():
            yield model
            return

        original_device = next(model.parameters()).device
        model.to(self.gpu_device)
        try:
            yield model
        finally:
            model.to(original_device)
            if original_device.type == "cpu":
                torch.cuda.empty_cache()
