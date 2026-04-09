"""
Simple metric logger: prints to stdout and writes to CSV.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Optional


class MetricLogger:
    """Accumulate and log training metrics."""

    def __init__(self, output_dir: str, filename: str = "metrics.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, filename)
        self._rows: list[dict] = []
        self._step_buffer: dict = {}

    def log(self, step: int, metrics: dict, print_msg: bool = True):
        row = {"step": step, **metrics}
        self._rows.append(row)
        if print_msg:
            parts = [f"step={step}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(" | ".join(parts))

    def flush(self):
        """Write all accumulated rows to CSV."""
        if not self._rows:
            return
        keys = list(self._rows[0].keys())
        # Gather all keys across rows
        for r in self._rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._rows)
