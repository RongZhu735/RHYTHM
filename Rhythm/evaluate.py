from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .config import RhythmConfig
from .data import InteractionDataset
from .sampler import time_unit_count
from .segmenter import IntentAwareSegmenter


def _intent_id_sequence_batch(
    users: list[int],
    train: dict[int, list[int]],
    train_times: dict[int, list[int]],
    maxlen: int,
    time_type: str,
) -> np.ndarray:
    time_units = time_unit_count(time_type)
    batch = np.zeros([len(users), time_units, maxlen], dtype=np.int32)
    for batch_idx, user in enumerate(users):
        counts = [0] * time_units
        for item, time_value in zip(train[user], train_times.get(user, [])):
            if 0 <= time_value < time_units and counts[time_value] < maxlen:
                batch[batch_idx, time_value, counts[time_value]] = item
                counts[time_value] += 1
    return batch


def evaluate_split(
    model,
    dataset: InteractionDataset,
    args: RhythmConfig,
    split: str,
    intent_adapter: IntentAwareSegmenter,
) -> tuple[float, float, float, float, float, float, float]:
    train = dataset.user_train
    valid = dataset.user_valid
    test = dataset.user_test
    train_times = dataset.user_train_times
    valid_times = dataset.user_valid_times
    test_times = dataset.user_test_times
    device = args.device

    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")

    target_dict = valid if split == "valid" else test
    valid_users = [
        user
        for user in range(1, dataset.usernum + 1)
        if user in train and user in target_dict and len(train[user]) >= 1 and len(target_dict[user]) >= 1
    ]

    ndcg5 = ndcg10 = ndcg20 = 0.0
    hr5 = hr10 = hr20 = 0.0
    mrr = 0.0
    valid_user_count = 0.0
    model.eval()

    with torch.no_grad():
        for start in tqdm(
            range(0, len(valid_users), args.eval_batch_size),
            desc=f"Evaluating {split} set",
            unit="batch",
            ncols=100,
        ):
            batch_users = valid_users[start : start + args.eval_batch_size]
            batch_size = len(batch_users)
            batch_seq = np.zeros([batch_size, args.maxlen], dtype=np.int32)
            batch_times = np.zeros([batch_size, args.maxlen], dtype=np.int32)
            batch_target_indices: list[int] = []

            for idx, user in enumerate(batch_users):
                seq_idx = args.maxlen - 1
                if split == "test":
                    batch_seq[idx, seq_idx] = valid[user][0]
                    if valid_times.get(user):
                        batch_times[idx, seq_idx] = valid_times[user][0]
                    seq_idx -= 1

                for rev_idx, item in enumerate(reversed(train[user])):
                    if seq_idx == -1:
                        break
                    batch_seq[idx, seq_idx] = item
                    user_times = train_times.get(user, [])
                    if len(user_times) > rev_idx:
                        batch_times[idx, seq_idx] = user_times[-rev_idx - 1]
                    seq_idx -= 1

                target_item = target_dict[user][0]
                batch_target_indices.append(target_item - 1)

            seq_tensor = torch.tensor(batch_seq, dtype=torch.int32, device=device)
            times_tensor = torch.tensor(batch_times, dtype=torch.int32, device=device)
            intent_ids = _intent_id_sequence_batch(batch_users, train, train_times, args.maxlen, args.time_type)
            intent_tensor = torch.from_numpy(intent_ids).to(device)
            segments = intent_adapter.process_intent_id_sequence(
                intent_tensor,
                use_fixed=args.use_fixed,
                K_max=args.kmax,
                min_delta=args.min_delta,
            )

            all_items = np.arange(1, dataset.itemnum + 1, dtype=np.int32)
            item_idx_array = np.asarray([all_items] * batch_size, dtype=np.int32)
            item_idx_tensor = torch.tensor(item_idx_array, dtype=torch.int32, device=device)
            predictions = -model.predict(seq_tensor, item_idx_tensor, times_tensor, segments).cpu().numpy()

            for idx, pred in enumerate(predictions):
                rank = pred.argsort().argsort()[batch_target_indices[idx]]
                valid_user_count += 1
                if rank < 5:
                    hr5 += 1.0
                    ndcg5 += 1 / np.log2(rank + 2)
                if rank < 10:
                    hr10 += 1.0
                    ndcg10 += 1 / np.log2(rank + 2)
                if rank < 20:
                    hr20 += 1.0
                    ndcg20 += 1 / np.log2(rank + 2)
                mrr += 1 / (rank + 1)

    if valid_user_count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        ndcg5 / valid_user_count,
        ndcg10 / valid_user_count,
        ndcg20 / valid_user_count,
        hr5 / valid_user_count,
        hr10 / valid_user_count,
        hr20 / valid_user_count,
        mrr / valid_user_count,
    )


def evaluate_valid(model, dataset: InteractionDataset, args: RhythmConfig, intent_adapter: IntentAwareSegmenter):
    return evaluate_split(model, dataset, args, "valid", intent_adapter)


def evaluate(model, dataset: InteractionDataset, args: RhythmConfig, intent_adapter: IntentAwareSegmenter):
    return evaluate_split(model, dataset, args, "test", intent_adapter)
