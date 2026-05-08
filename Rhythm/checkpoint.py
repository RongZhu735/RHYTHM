from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CheckpointReport:
    loaded: list[str]
    missing: list[str]
    unexpected: list[str]
    shape_mismatch: list[str]

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded)}, missing={len(self.missing)}, "
            f"unexpected={len(self.unexpected)}, shape_mismatch={len(self.shape_mismatch)}"
        )


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def load_legacy_checkpoint(model, checkpoint_path: str, map_location="cpu") -> CheckpointReport:
    """Load checkpoints by matching key names and tensor shapes."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _unwrap_state_dict(checkpoint)
    model_state = model.state_dict()

    compatible = {}
    unexpected = []
    shape_mismatch = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if hasattr(value, "shape") and model_state[key].shape != value.shape:
            shape_mismatch.append(key)
            continue
        compatible[key] = value

    missing = [key for key in model_state if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    return CheckpointReport(
        loaded=sorted(compatible),
        missing=sorted(missing),
        unexpected=sorted(unexpected),
        shape_mismatch=sorted(shape_mismatch),
    )
