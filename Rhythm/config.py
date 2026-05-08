from __future__ import annotations

from dataclasses import dataclass


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value not in {"False", "True", "false", "true", "0", "1"}:
        raise ValueError(f"Not a valid boolean string: {value}")
    return value in {"True", "true", "1"}


@dataclass
class RhythmConfig:
    dataset: str
    train_dir: str
    batch_size: int = 512
    lr: float = 1e-3
    maxlen: int = 50
    hidden_units: int = 128
    num_blocks: int = 1
    num_epochs: int = 300
    stop: int = 10
    num_heads: int = 4
    attn_dropout_rate: float = 0.5
    ff_dropout_rate: float = 0.5
    l2_emb: float = 0.0
    device: str = "auto"
    eval_batch_size: int = 256
    num_workers: int = 1
    use_cross: bool = True
    use_cross_withfnn: bool = False
    learnable_intent: bool = False
    loss_type: str = "full_ce"
    n_negatives: int = 256
    time_type: str = "hour"
    use_fixed: bool = True
    kmax: int = 4
    min_delta: float = 0.1
    step: int = 2
    seed: int = 2026
    output_dir: str | None = None
    checkpoint: str | None = None
    eval_only: bool = False

    # Fixed branches from the original paper code. They are kept as attributes so
    # older helper code and checkpoints can still reason about the chosen mode.
    use_time_mask_train: bool = True
    use_time_mask_eval: bool = True
    use_causal_mask_train: bool = True
    use_causal_mask_eval: bool = True
    use_llm_embedding: bool = False
    write_attention_weights: bool = False

    @property
    def resolved_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        return f"{self.dataset}_{self.train_dir}"

    @property
    def use_full_ce(self) -> bool:
        return self.loss_type in {"ce", "full_ce"}

    def as_args_dict(self) -> dict[str, object]:
        return dict(sorted(self.__dict__.items(), key=lambda kv: kv[0]))
