from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict


@dataclass
class InteractionDataset:
    user_train: dict[int, list[int]]
    user_valid: dict[int, list[int]]
    user_test: dict[int, list[int]]
    usernum: int
    itemnum: int
    item_embedding_info: dict[int, str]
    user_train_times: dict[int, list[int]]
    user_valid_times: dict[int, list[int]]
    user_test_times: dict[int, list[int]]

    def as_legacy_tuple(self):
        return [
            self.user_train,
            self.user_valid,
            self.user_test,
            self.usernum,
            self.itemnum,
            self.item_embedding_info,
            self.user_train_times,
            self.user_valid_times,
            self.user_test_times,
        ]

    @property
    def average_train_length(self) -> float:
        if not self.user_train:
            return 0.0
        return sum(len(seq) for seq in self.user_train.values()) / len(self.user_train)


def _normalized_fieldnames(fieldnames: list[str]) -> dict[str, str]:
    return {name.split(":", 1)[0]: name for name in fieldnames}


def _detect_delimiter(path: str) -> str:
    if path.endswith(".inter") or path.endswith(".tsv"):
        return "\t"
    with open(path, "r", encoding="utf-8") as fp:
        sample = fp.read(4096)
    return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter


def _pick_column(fields: dict[str, str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    for name in candidates:
        if name in fields:
            return fields[name]
    if required:
        raise ValueError(f"Dataset is missing required column. Expected one of: {', '.join(candidates)}")
    return None


def _time_column(fields: dict[str, str], preferred_time_type: str | None) -> str | None:
    if preferred_time_type and preferred_time_type in fields:
        return fields[preferred_time_type]
    for name in ("hour", "month", "day_of_week"):
        if name in fields:
            return fields[name]
    return None


def load_dataset(path: str, time_type: str | None = None, sort_by_timestamp: bool = True) -> InteractionDataset:
    """Load a CSV/TSV/RecBole .inter sequential recommendation dataset.

    Required columns are `user_id` and either `product_id` or `item_id`.
    Optional time columns are `hour`, `month`, or `day_of_week`; missing values
    are treated as 0, matching the legacy code's fallback.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    delimiter = _detect_delimiter(path)
    user_items: DefaultDict[int, list[tuple[int, int, int]]] = defaultdict(list)
    item_embedding_info: dict[int, str] = {}
    usernum = 0
    itemnum = 0

    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Dataset has no header: {path}")

        fields = _normalized_fieldnames(reader.fieldnames)
        user_col = _pick_column(fields, ("user_id", "uid", "user"))
        item_col = _pick_column(fields, ("product_id", "item_id", "iid", "item"))
        timestamp_col = _pick_column(fields, ("timestamp", "time"), required=False)
        time_col = _time_column(fields, time_type)
        info_col = _pick_column(fields, ("item_embedding_info",), required=False)

        for order, row in enumerate(reader):
            user = int(float(row[user_col]))
            item = int(float(row[item_col]))
            raw_time = row[time_col] if time_col is not None and row.get(time_col, "") != "" else 0
            time_value = int(float(raw_time))
            timestamp = int(float(row[timestamp_col])) if timestamp_col and row.get(timestamp_col, "") != "" else order

            usernum = max(usernum, user)
            itemnum = max(itemnum, item)
            user_items[user].append((timestamp, order, item, time_value))
            if info_col and row.get(info_col):
                item_embedding_info[item] = row[info_col]

    user_train: dict[int, list[int]] = {}
    user_valid: dict[int, list[int]] = {}
    user_test: dict[int, list[int]] = {}
    user_train_times: dict[int, list[int]] = {}
    user_valid_times: dict[int, list[int]] = {}
    user_test_times: dict[int, list[int]] = {}

    for user, interactions in user_items.items():
        if sort_by_timestamp:
            interactions = sorted(interactions, key=lambda value: (value[0], value[1]))

        items = [item for _, _, item, _ in interactions]
        times = [time_value for _, _, _, time_value in interactions]
        nfeedback = len(items)

        if nfeedback < 3:
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
            user_train_times[user] = times
            user_valid_times[user] = []
            user_test_times[user] = []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]
            user_train_times[user] = times[:-2]
            user_valid_times[user] = [times[-2]]
            user_test_times[user] = [times[-1]]

    return InteractionDataset(
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        usernum=usernum,
        itemnum=itemnum,
        item_embedding_info=item_embedding_info,
        user_train_times=user_train_times,
        user_valid_times=user_valid_times,
        user_test_times=user_test_times,
    )


def data_partition(path: str):
    """Legacy-compatible wrapper around `load_dataset`."""
    return load_dataset(path).as_legacy_tuple()
