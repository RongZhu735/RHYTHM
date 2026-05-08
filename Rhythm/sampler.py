from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


def time_unit_count(time_type: str) -> int:
    if time_type == "hour":
        return 24
    if time_type == "month":
        return 12
    if time_type == "day_of_week":
        return 7
    raise ValueError(f"Unsupported time_type: {time_type}")


@dataclass
class Batch:
    users: np.ndarray
    seq: np.ndarray
    pos: np.ndarray
    neg: np.ndarray | None
    times: np.ndarray
    intent_id_sequence: np.ndarray


class SequenceBatchSampler:
    """Single-process sampler for SASRec-style next-item training."""

    def __init__(
        self,
        user_train: dict[int, list[int]],
        user_train_times: dict[int, list[int]],
        usernum: int,
        itemnum: int,
        batch_size: int,
        maxlen: int,
        n_negatives: int = 1,
        time_type: str = "hour",
        use_full_ce: bool = True,
        seed: int | None = None,
    ):
        self.user_train = user_train
        self.user_train_times = user_train_times
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.n_negatives = n_negatives
        self.time_type = time_type
        self.use_full_ce = use_full_ce
        self.time_units = time_unit_count(time_type)
        self.rng = np.random.default_rng(seed)
        self.python_rng = random.Random(seed)
        self.available_users = [u for u in range(1, usernum + 1) if len(user_train.get(u, [])) > 1]
        if not self.available_users:
            raise ValueError("No users have at least two training interactions.")

    def _negative_sample(self, history: set[int], size: tuple[int, ...]) -> np.ndarray:
        result = np.zeros(size, dtype=np.int32)
        flat = result.reshape(-1)
        for idx in range(flat.size):
            item = self.python_rng.randint(1, self.itemnum)
            while item in history:
                item = self.python_rng.randint(1, self.itemnum)
            flat[idx] = item
        return result

    def _sample_one(self):
        user = int(self.rng.choice(self.available_users))
        items = self.user_train[user]
        times = self.user_train_times.get(user, [0] * len(items))

        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        seq_times = np.zeros([self.maxlen], dtype=np.int32)
        nxt = items[-1]
        idx = self.maxlen - 1

        for item_idx in reversed(range(len(items) - 1)):
            seq[idx] = items[item_idx]
            pos[idx] = nxt
            if item_idx < len(times):
                seq_times[idx] = times[item_idx]
            nxt = items[item_idx]
            idx -= 1
            if idx == -1:
                break

        intent_id_sequence = np.zeros([self.time_units, self.maxlen], dtype=np.int32)
        time_item_counts = [0] * self.time_units
        for item, time_value in zip(items[:-1], times[: len(items) - 1]):
            if 0 <= time_value < self.time_units and time_item_counts[time_value] < self.maxlen:
                intent_id_sequence[time_value, time_item_counts[time_value]] = item
                time_item_counts[time_value] += 1

        if self.use_full_ce:
            neg = None
        else:
            valid_positions = pos != 0
            history = set(items)
            if self.n_negatives == 1:
                neg = np.zeros([self.maxlen], dtype=np.int32)
                neg[valid_positions] = self._negative_sample(history, (int(valid_positions.sum()),))
            else:
                neg = np.zeros([self.maxlen, self.n_negatives], dtype=np.int32)
                neg[valid_positions] = self._negative_sample(history, (int(valid_positions.sum()), self.n_negatives))

        return user, seq, pos, neg, seq_times, intent_id_sequence

    def next_batch(self) -> Batch:
        samples = [self._sample_one() for _ in range(self.batch_size)]
        users, seq, pos, neg, times, intent_id_sequence = zip(*samples)
        batch_neg = None if self.use_full_ce else np.stack(neg, axis=0)
        return Batch(
            users=np.asarray(users, dtype=np.int32),
            seq=np.stack(seq, axis=0),
            pos=np.stack(pos, axis=0),
            neg=batch_neg,
            times=np.stack(times, axis=0),
            intent_id_sequence=np.stack(intent_id_sequence, axis=0),
        )

    def close(self) -> None:
        return None


# Backward-compatible alias retained for external scripts.
WarpSampler = SequenceBatchSampler
