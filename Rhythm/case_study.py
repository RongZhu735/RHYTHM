from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Rhythm"

from .checkpoint import load_legacy_checkpoint
from .config import RhythmConfig
from .data import InteractionDataset, load_dataset
from .evaluate import _intent_id_sequence_batch
from .main import choose_device, seed_everything
from .model import SASRec
from .sampler import time_unit_count
from .segmenter import IntentAwareSegmenter


PROJECT_DIR = Path(__file__).resolve().parents[1]
RHYTHM_DIR = Path(__file__).resolve().parent

DATASET_SPECS = {
    "DHRD": {
        "args": RHYTHM_DIR / "model" / "DHRD" / "args.txt",
        "data": RHYTHM_DIR / "data" / "DHRD.csv",
        "checkpoint": RHYTHM_DIR / "model" / "DHRD" / "best_model.pt",
    },
    "megamarket": {
        "args": RHYTHM_DIR / "model" / "megamarket" / "args.txt",
        "data": RHYTHM_DIR / "data" / "megamarket.csv",
        "checkpoint": RHYTHM_DIR / "model" / "megamarket" / "best_model.pt",
    },
    "ta-feng": {
        "args": RHYTHM_DIR / "model" / "ta-feng" / "args.txt",
        "data": RHYTHM_DIR / "data" / "ta-feng.csv",
        "checkpoint": RHYTHM_DIR / "model" / "ta-feng" / "best_model.pt",
    },
}


def read_args_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(",", 1)
        values[key] = value
    return values


def coerce_config_values(raw_values: dict[str, str]) -> dict[str, object]:
    bool_fields = {
        "eval_only",
        "learnable_intent",
        "use_cross",
        "use_cross_withfnn",
        "use_fixed",
        "use_time_mask_train",
        "use_time_mask_eval",
        "use_causal_mask_train",
        "use_causal_mask_eval",
        "use_llm_embedding",
        "write_attention_weights",
    }
    int_fields = {
        "batch_size",
        "eval_batch_size",
        "hidden_units",
        "kmax",
        "maxlen",
        "n_negatives",
        "num_blocks",
        "num_epochs",
        "num_heads",
        "num_workers",
        "seed",
        "step",
        "stop",
    }
    float_fields = {"attn_dropout_rate", "ff_dropout_rate", "l2_emb", "lr", "min_delta"}

    known_fields = {field.name for field in fields(RhythmConfig)}
    values: dict[str, object] = {}
    for key, value in raw_values.items():
        if key not in known_fields:
            continue
        if key in bool_fields:
            values[key] = value in {"True", "true", "1"}
        elif key in int_fields:
            values[key] = int(value)
        elif key in float_fields:
            values[key] = float(value)
        elif value == "None":
            values[key] = None
        else:
            values[key] = value
    return values


def build_config(dataset_name: str, device: str, eval_batch_size: int | None = None) -> tuple[RhythmConfig, Path]:
    spec = DATASET_SPECS[dataset_name]
    raw_values = read_args_file(spec["args"])
    config_values = coerce_config_values(raw_values)
    config_values["dataset"] = str(spec["data"])
    config_values["train_dir"] = f"case_study_{dataset_name}"
    config_values["checkpoint"] = str(spec["checkpoint"])
    config_values["device"] = device
    config_values["eval_only"] = True
    if eval_batch_size is not None:
        config_values["eval_batch_size"] = eval_batch_size
    return RhythmConfig(**config_values), spec["checkpoint"]


def valid_users_for_split(dataset: InteractionDataset, split: str) -> list[int]:
    target = dataset.user_valid if split == "valid" else dataset.user_test
    return [
        user
        for user in range(1, dataset.usernum + 1)
        if user in dataset.user_train and user in target and len(dataset.user_train[user]) >= 1 and len(target[user]) >= 1
    ]


def user_temporal_stats(dataset: InteractionDataset, args: RhythmConfig, user: int) -> dict[str, object]:
    time_units = time_unit_count(args.time_type)
    train_times = [
        int(time_value)
        for time_value in dataset.user_train_times.get(user, [])
        if 0 <= int(time_value) < time_units
    ]
    active_time_units = sorted(set(train_times))
    time_unit_span = 0
    if active_time_units:
        time_unit_span = active_time_units[-1] - active_time_units[0] + 1

    return {
        "time_unit_span": int(time_unit_span),
        "active_time_unit_count": int(len(active_time_units)),
        "history_item_count": int(len(dataset.user_train.get(user, []))),
        "active_time_units": active_time_units,
    }


def temporal_selection_key(stats: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(stats["time_unit_span"]),
        int(stats["active_time_unit_count"]),
        int(stats["history_item_count"]),
    )


def build_eval_sequence(dataset: InteractionDataset, args: RhythmConfig, user: int, split: str):
    seq = np.zeros([args.maxlen], dtype=np.int32)
    times = np.zeros([args.maxlen], dtype=np.int32)
    seq_idx = args.maxlen - 1

    if split == "test":
        seq[seq_idx] = dataset.user_valid[user][0]
        if dataset.user_valid_times.get(user):
            times[seq_idx] = dataset.user_valid_times[user][0]
        seq_idx -= 1

    user_times = dataset.user_train_times.get(user, [])
    for rev_idx, item in enumerate(reversed(dataset.user_train[user])):
        if seq_idx == -1:
            break
        seq[seq_idx] = item
        if len(user_times) > rev_idx:
            times[seq_idx] = user_times[-rev_idx - 1]
        seq_idx -= 1

    target = dataset.user_valid[user][0] if split == "valid" else dataset.user_test[user][0]
    return seq, times, target


def collect_ranked_cases(
    model: SASRec,
    dataset: InteractionDataset,
    args: RhythmConfig,
    intent_adapter: IntentAwareSegmenter,
    split: str,
    rank_threshold: int,
    max_scan_users: int,
    user_selection: str,
    random_pool_size: int,
    seed: int,
) -> list[dict[str, object]]:
    users = valid_users_for_split(dataset, split)
    user_stats = {int(user): user_temporal_stats(dataset, args, int(user)) for user in users}
    if user_selection == "random":
        rng = np.random.default_rng(seed)
        users = list(rng.permutation(users))
        if max_scan_users <= 0:
            max_scan_users = max(random_pool_size, args.eval_batch_size, 1)
    elif user_selection == "coverage":
        users = sorted(
            users,
            key=lambda user: (
                *temporal_selection_key(user_stats[int(user)]),
                -int(user),
            ),
            reverse=True,
        )
    elif user_selection != "rank":
        raise ValueError(f"Unsupported user_selection: {user_selection}")

    if max_scan_users > 0:
        users = users[:max_scan_users]
    users = [int(user) for user in users]

    all_items = np.arange(1, dataset.itemnum + 1, dtype=np.int32)
    cases: list[dict[str, object]] = []
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, len(users), args.eval_batch_size), desc=f"Scanning {split} cases", ncols=100):
            batch_users = users[start : start + args.eval_batch_size]
            batch_seq = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
            batch_times = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
            targets = []

            for idx, user in enumerate(batch_users):
                seq, times, target = build_eval_sequence(dataset, args, user, split)
                batch_seq[idx] = seq
                batch_times[idx] = times
                targets.append(target)

            seq_tensor = torch.tensor(batch_seq, dtype=torch.int32, device=args.device)
            times_tensor = torch.tensor(batch_times, dtype=torch.int32, device=args.device)
            intent_ids = _intent_id_sequence_batch(
                batch_users,
                dataset.user_train,
                dataset.user_train_times,
                args.maxlen,
                args.time_type,
            )
            intent_tensor = torch.from_numpy(intent_ids).to(args.device)
            time_unit_segments = intent_adapter.process_intent_id_sequence(
                intent_tensor,
                use_fixed=args.use_fixed,
                K_max=args.kmax,
                min_delta=args.min_delta,
            )

            item_idx_array = np.asarray([all_items] * len(batch_users), dtype=np.int32)
            item_idx_tensor = torch.tensor(item_idx_array, dtype=torch.int32, device=args.device)
            logits = model.predict(seq_tensor, item_idx_tensor, times_tensor, time_unit_segments)
            logits_np = logits.detach().cpu().numpy()

            for idx, user in enumerate(batch_users):
                target = targets[idx]
                target_score = logits_np[idx, target - 1]
                rank = int(np.sum(logits_np[idx] > target_score))
                if rank >= rank_threshold:
                    continue
                top_indices = np.argsort(-logits_np[idx])[:10]
                cases.append(
                    {
                        "user": user,
                        "rank": rank,
                        "target": target,
                        "target_score": float(target_score),
                        "top_items": (top_indices + 1).astype(int).tolist(),
                        "top_scores": logits_np[idx, top_indices].astype(float).tolist(),
                        "seq": batch_seq[idx].astype(int).tolist(),
                        "times": batch_times[idx].astype(int).tolist(),
                        "intent_id_sequence": intent_ids[idx].astype(int).tolist(),
                        "time_unit_segments": time_unit_segments[idx].detach().cpu().long().tolist(),
                        "time_unit_span": user_stats[int(user)]["time_unit_span"],
                        "active_time_unit_count": user_stats[int(user)]["active_time_unit_count"],
                        "history_item_count": user_stats[int(user)]["history_item_count"],
                        "history_active_time_units": user_stats[int(user)]["active_time_units"],
                    }
                )

    if user_selection == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(cases)
    elif user_selection == "coverage":
        cases.sort(
            key=lambda item: (
                -int(item["time_unit_span"]),
                -int(item["active_time_unit_count"]),
                -int(item["history_item_count"]),
                int(item["rank"]),
                -float(item["target_score"]),
                int(item["user"]),
            )
        )
    else:
        cases.sort(key=lambda item: (int(item["rank"]), -float(item["target_score"]), int(item["user"])))
    return cases


def analyze_case(model: SASRec, args: RhythmConfig, case: dict[str, object]) -> dict[str, object]:
    seq = torch.tensor([case["seq"]], dtype=torch.int32, device=args.device)
    times = torch.tensor([case["times"]], dtype=torch.int32, device=args.device)
    time_unit_segments = torch.tensor([case["time_unit_segments"]], dtype=torch.long, device=args.device)

    _, seq_segments = model.generate_time_mask(times, use_time_mask=True, segments=time_unit_segments)
    seq_segments = seq_segments[0]
    item_emb_table = model.embedding.embedding_table.weight
    target = int(case["target"])
    target_vec = item_emb_table[target]
    intent_id_sequence = torch.tensor(case["intent_id_sequence"], dtype=torch.long, device=args.device)
    time_unit_segments_1d = time_unit_segments[0]
    time_unit_item_counts = (intent_id_sequence != 0).sum(dim=1).detach().cpu().long().tolist()
    active_time_units = [idx for idx, count in enumerate(time_unit_item_counts) if int(count) > 0]

    segment_stats = []
    for segment_id in sorted(time_unit_segments_1d.detach().cpu().unique().tolist()):
        unit_mask = time_unit_segments_1d == int(segment_id)
        segment_items_by_unit = intent_id_sequence[unit_mask]
        segment_items = segment_items_by_unit.reshape(-1)
        segment_items = segment_items[segment_items != 0]
        segment_times = []
        for unit_idx in torch.nonzero(unit_mask, as_tuple=False).flatten().detach().cpu().long().tolist():
            unit_items = intent_id_sequence[unit_idx]
            non_zero_count = int((unit_items != 0).sum().item())
            segment_times.extend([unit_idx] * non_zero_count)

        similarity = None
        if segment_items.numel() > 0:
            segment_vec = item_emb_table[segment_items].mean(dim=0)
            similarity = F.cosine_similarity(segment_vec.unsqueeze(0), target_vec.unsqueeze(0)).item()
        time_units = [
            idx
            for idx, value in enumerate(case["time_unit_segments"])
            if int(value) == int(segment_id)
        ]
        segment_active_time_units = [idx for idx in time_units if int(time_unit_item_counts[idx]) > 0]
        segment_stats.append(
            {
                "segment_id": int(segment_id),
                "similarity": None if similarity is None else float(similarity),
                "item_count": int(segment_items.numel()),
                "items": segment_items.detach().cpu().long().tolist(),
                "times": segment_times,
                "time_units": time_units,
                "active_time_units": segment_active_time_units,
            }
        )

    analyzed = dict(case)
    analyzed["seq_segments"] = seq_segments.detach().cpu().long().tolist()
    analyzed["segments"] = segment_stats
    analyzed["time_unit_item_counts"] = [int(count) for count in time_unit_item_counts]
    analyzed["active_time_units"] = active_time_units
    return analyzed


def write_case_tables(dataset_name: str, cases: list[dict[str, object]], output_dir: Path) -> None:
    summary_path = output_dir / f"{dataset_name}_cases_summary.csv"
    segment_path = output_dir / f"{dataset_name}_segment_similarity.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "dataset",
                "case_id",
                "user",
                "split",
                "target",
                "rank",
                "time_unit_span",
                "active_time_unit_count",
                "history_item_count",
                "history_active_time_units",
                "top_items",
                "top_scores",
            ],
        )
        writer.writeheader()
        for case_id, case in enumerate(cases, start=1):
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "user": case["user"],
                    "split": case["split"],
                    "target": case["target"],
                    "rank": case["rank"],
                    "time_unit_span": case["time_unit_span"],
                    "active_time_unit_count": case["active_time_unit_count"],
                    "history_item_count": case["history_item_count"],
                    "history_active_time_units": json.dumps(case["history_active_time_units"], ensure_ascii=False),
                    "top_items": json.dumps(case["top_items"], ensure_ascii=False),
                    "top_scores": json.dumps(case["top_scores"], ensure_ascii=False),
                }
            )

    with segment_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "dataset",
                "case_id",
                "user",
                "target",
                "rank",
                "segment_id",
                "similarity",
                "item_count",
                "items",
                "times",
                "time_units",
                "active_time_units",
            ],
        )
        writer.writeheader()
        for case_id, case in enumerate(cases, start=1):
            for segment in sorted_segments(case):
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "case_id": case_id,
                        "user": case["user"],
                        "target": case["target"],
                        "rank": case["rank"],
                        "segment_id": segment["segment_id"],
                        "similarity": "" if segment["similarity"] is None else f"{segment['similarity']:.6f}",
                        "item_count": segment["item_count"],
                        "items": json.dumps(segment["items"], ensure_ascii=False),
                        "times": json.dumps(segment["times"], ensure_ascii=False),
                        "time_units": json.dumps(segment["time_units"], ensure_ascii=False),
                        "active_time_units": json.dumps(segment["active_time_units"], ensure_ascii=False),
                    }
                )


def sorted_segments(case: dict[str, object]) -> list[dict[str, object]]:
    active_segments = [
        segment
        for segment in case["segments"]
        if int(segment["item_count"]) > 0 and segment["similarity"] is not None
    ]

    def sort_key(segment: dict[str, object]) -> tuple[bool, float]:
        similarity = segment["similarity"]
        if similarity is None:
            return False, -float("inf")
        return int(segment["item_count"]) > 0, float(similarity)

    return sorted(active_segments, key=sort_key, reverse=True)


def time_unit_item_counts(case: dict[str, object], expected_units: int) -> np.ndarray:
    counts = np.asarray(case.get("time_unit_item_counts", []), dtype=int)
    if counts.size == 0:
        intent_id_sequence = np.asarray(case.get("intent_id_sequence", []), dtype=int)
        if intent_id_sequence.ndim == 2:
            counts = (intent_id_sequence != 0).sum(axis=1).astype(int)
        elif intent_id_sequence.ndim == 1:
            counts = (intent_id_sequence != 0).astype(int)

    if counts.size < expected_units:
        counts = np.pad(counts, (0, expected_units - counts.size), constant_values=0)
    elif counts.size > expected_units:
        counts = counts[:expected_units]
    return counts


def draw_time_unit_ring(ax, case: dict[str, object], args: RhythmConfig, case_id: int) -> None:
    time_segments = np.asarray(case["time_unit_segments"], dtype=int)
    time_units = len(time_segments)
    if time_units == 0:
        ax.axis("off")
        return
    unit_item_counts = time_unit_item_counts(case, time_units)

    best_segment = None
    for segment in sorted_segments(case):
        if segment["similarity"] is not None and int(segment["item_count"]) > 0:
            best_segment = segment["segment_id"]
            break
    segment_palette = {
        0: "#f4a261",
        1: "#e9c46a",
        2: "#2a9d8f",
        3: "#457b9d",
        4: "#8d99ae",
        5: "#b56576",
        6: "#6a994e",
        7: "#9d4edd",
    }

    theta = np.linspace(0.0, 2 * np.pi, time_units, endpoint=False)
    width = (2 * np.pi / time_units) * 0.995
    colors = []
    edge_colors = []
    line_widths = []
    for unit_idx, segment_id in enumerate(time_segments):
        if int(unit_item_counts[unit_idx]) == 0:
            colors.append("#e6e6e6")
            edge_colors.append("#ffffff")
            line_widths.append(0.45)
        elif best_segment is not None and int(segment_id) == int(best_segment):
            colors.append("#c1121f")
            edge_colors.append("#5c0007")
            line_widths.append(1.2)
        else:
            colors.append(segment_palette[int(segment_id) % len(segment_palette)])
            edge_colors.append("white")
            line_widths.append(0.55)

    ax.bar(theta, np.ones(time_units) * 0.34, width=width, bottom=0.56, color=colors, edgecolor=edge_colors, linewidth=line_widths)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.text(
        -0.18,
        0.5,
        f"C{case_id}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
    )


def plot_dataset_overview(dataset_name: str, cases: list[dict[str, object]], args: RhythmConfig, output_dir: Path) -> None:
    """Draw one fused figure for multiple cases.

    Rows are cases. Columns are active segments sorted by target similarity
    within each case. Empty time units are intentionally excluded from the
    explanation so the figure focuses on temporal units that contain history.
    """
    if not cases:
        return

    sorted_by_case = [sorted_segments(case) for case in cases]
    max_segments = max(len(segments) for segments in sorted_by_case)
    if max_segments == 0:
        return
    similarities = np.full((len(cases), max_segments), np.nan, dtype=float)
    annotations = np.full((len(cases), max_segments), "", dtype=object)

    for row_idx, segments in enumerate(sorted_by_case):
        for col_idx, segment in enumerate(segments):
            if segment["similarity"] is None:
                annotations[row_idx, col_idx] = f"S{segment['segment_id']}\n--"
            else:
                similarities[row_idx, col_idx] = float(segment["similarity"])
                annotations[row_idx, col_idx] = (
                    f"S{segment['segment_id']}\n"
                    f"{segment['similarity']:.2f}"
                )

    finite_values = similarities[np.isfinite(similarities)]
    if finite_values.size == 0:
        return
    color_min = min(0.0, float(finite_values.min()))
    color_max = max(0.1, float(finite_values.max()))

    height = max(4.8, 1.18 * len(cases) + 2.3)
    width = max(10.8, 1.55 * max_segments + 5.1)
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(
        len(cases),
        2,
        width_ratios=[max(4.5, 1.15 * max_segments), 1.85],
        wspace=0.2,
        hspace=0.68,
    )
    heatmap_ax = fig.add_subplot(grid[:, 0])
    ring_axes = [fig.add_subplot(grid[row_idx, 1], projection="polar") for row_idx in range(len(cases))]

    masked = np.ma.masked_invalid(similarities)
    image = heatmap_ax.imshow(masked, aspect="auto", cmap="Reds", vmin=color_min, vmax=color_max)
    heatmap_ax.set_title("Active segment-target similarity across selected cases", fontsize=12, fontweight="bold")
    heatmap_ax.set_xlabel("Active discovered segments sorted by similarity to target")
    heatmap_ax.set_xticks(range(max_segments))
    heatmap_ax.set_xticklabels([f"Top-{idx + 1}" for idx in range(max_segments)])
    heatmap_ax.set_yticks(range(len(cases)))
    heatmap_ax.set_yticklabels(
        [
            f"C{idx + 1}: user{case['user']}"
            for idx, case in enumerate(cases)
        ],
        fontsize=8,
    )

    for row_idx in range(len(cases)):
        for col_idx in range(max_segments):
            if not np.isfinite(similarities[row_idx, col_idx]):
                if annotations[row_idx, col_idx]:
                    heatmap_ax.text(
                        col_idx,
                        row_idx,
                        annotations[row_idx, col_idx],
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#555555",
                    )
                continue
            normalized_value = (similarities[row_idx, col_idx] - color_min) / max(color_max - color_min, 1e-8)
            heatmap_ax.text(
                col_idx,
                row_idx,
                annotations[row_idx, col_idx],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized_value > 0.54 else "#1f1f1f",
                fontweight="bold" if col_idx == 0 else "normal",
            )

    colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.02)
    colorbar.set_label("cosine(segment vector, target item vector)")

    for case_id, (case, ring_ax) in enumerate(zip(cases, ring_axes), start=1):
        draw_time_unit_ring(ring_ax, case, args, case_id)

    fig.suptitle(
        f"{dataset_name}: discovered segments most related to the target item "
        f"({args.time_type})",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.86, bottom=0.11, wspace=0.24, hspace=0.68)
    fig.savefig(output_dir / f"{dataset_name}_case_study_overview.png", dpi=240)
    plt.close(fig)


def run_dataset(dataset_name: str, cli_args: argparse.Namespace) -> None:
    args, checkpoint_path = build_config(dataset_name, cli_args.device, cli_args.eval_batch_size)
    args.device = choose_device(args.device)
    dataset_output_dir = Path(cli_args.output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, time_type=args.time_type)
    model = SASRec(dataset.usernum, dataset.itemnum, args).to(args.device)
    report = load_legacy_checkpoint(model, str(checkpoint_path), map_location=args.device)
    print(f"[{dataset_name}] checkpoint: {report.summary()}")
    model.eval()
    intent_adapter = IntentAwareSegmenter(args.hidden_units, model.embedding.embedding_table).to(args.device)

    ranked_cases = collect_ranked_cases(
        model=model,
        dataset=dataset,
        args=args,
        intent_adapter=intent_adapter,
        split=cli_args.split,
        rank_threshold=cli_args.rank_threshold,
        max_scan_users=cli_args.max_scan_users,
        user_selection=cli_args.user_selection,
        random_pool_size=cli_args.random_pool_size,
        seed=cli_args.seed,
    )

    analyzed_cases = []
    for case in ranked_cases:
        analyzed = analyze_case(model, args, case)
        analyzed["dataset"] = dataset_name
        analyzed["split"] = cli_args.split
        if len(sorted_segments(analyzed)) < cli_args.min_segments:
            continue
        analyzed_cases.append(analyzed)
        if len(analyzed_cases) >= cli_args.max_cases:
            break

    if not analyzed_cases:
        print(f"[{dataset_name}] no cases found. Try increasing --rank-threshold or --max-scan-users.")
        return

    plot_dataset_overview(dataset_name, analyzed_cases, args, dataset_output_dir)

    for case_id, case in enumerate(analyzed_cases, start=1):
        with (dataset_output_dir / f"{dataset_name}_case_{case_id:02d}_user_{case['user']}.json").open("w", encoding="utf-8") as fp:
            json.dump(case, fp, ensure_ascii=False, indent=2)

    write_case_tables(dataset_name, analyzed_cases, dataset_output_dir)
    print(f"[{dataset_name}] wrote {len(analyzed_cases)} cases to {dataset_output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Rhythm segmentation case-study figures.")
    parser.add_argument("--dataset-name", default="all", choices=["all", *DATASET_SPECS.keys()])
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", default=None, type=int)
    parser.add_argument("--max-cases", default=5, type=int)
    parser.add_argument("--rank-threshold", default=10, type=int, help="Keep cases whose 0-based full ranking is below this threshold.")
    parser.add_argument("--max-scan-users", default=0, type=int, help="0 means scan all valid users.")
    parser.add_argument(
        "--user-selection",
        default="coverage",
        choices=["coverage", "rank", "random"],
        help="coverage prioritizes wide time-unit span and dense histories; rank sorts by prediction quality; random shuffles users before scanning.",
    )
    parser.add_argument("--random-pool-size", default=1000, type=int, help="Number of random users to scan when --user-selection random and --max-scan-users is 0.")
    parser.add_argument(
        "--min-segments",
        default=1,
        type=int,
        help="Minimum number of active segments, i.e. segments containing at least one history item.",
    )
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--output-dir", default=str(RHYTHM_DIR / "case_study_outputs"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    dataset_names = list(DATASET_SPECS) if args.dataset_name == "all" else [args.dataset_name]
    for dataset_name in dataset_names:
        run_dataset(dataset_name, args)


if __name__ == "__main__":
    main()
