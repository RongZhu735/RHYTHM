from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RHYTHM_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = RHYTHM_DIR / "case_study_outputs"

DATASETS = [
    ("DHRD", "DHRD", "hour"),
    ("megamarket", "MegaMarket", "day of week"),
    ("ta-feng", "Ta-Feng", "day of week"),
]

SEGMENT_PALETTE = {
    0: "#f4a261",
    1: "#e9c46a",
    2: "#2a9d8f",
    3: "#457b9d",
    4: "#8d99ae",
    5: "#b56576",
    6: "#6a994e",
    7: "#9d4edd",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def active_sorted_segments(case: dict[str, object]) -> list[dict[str, object]]:
    segments = [
        segment
        for segment in case["segments"]
        if segment["similarity"] is not None and int(segment["item_count"]) > 0
    ]
    return sorted(segments, key=lambda segment: float(segment["similarity"]), reverse=True)


def case_json_path(dataset_dir: Path, dataset_name: str, case_id: int, user: str) -> Path:
    exact_path = dataset_dir / f"{dataset_name}_case_{case_id:02d}_user_{user}.json"
    if exact_path.exists():
        return exact_path
    matches = sorted(dataset_dir.glob(f"{dataset_name}_case_{case_id:02d}_user_*.json"))
    if not matches:
        raise FileNotFoundError(f"Cannot find case JSON for {dataset_name} case {case_id}")
    return matches[-1]


def load_dataset_cases(base_dir: Path, dataset_name: str, max_cases: int | None) -> list[dict[str, object]]:
    dataset_dir = base_dir / dataset_name
    summary_rows = read_csv(dataset_dir / f"{dataset_name}_cases_summary.csv")
    if max_cases is not None:
        summary_rows = summary_rows[:max_cases]

    cases = []
    for row in summary_rows:
        case_id = int(row["case_id"])
        path = case_json_path(dataset_dir, dataset_name, case_id, row["user"])
        with path.open(encoding="utf-8") as fp:
            case = json.load(fp)
        case["case_id"] = case_id
        cases.append(case)
    return cases


def time_unit_item_counts(case: dict[str, object], expected_units: int) -> np.ndarray:
    counts = np.asarray(case.get("time_unit_item_counts", []), dtype=int)
    if counts.size < expected_units:
        counts = np.pad(counts, (0, expected_units - counts.size), constant_values=0)
    elif counts.size > expected_units:
        counts = counts[:expected_units]
    return counts


def draw_case_label(ax, case_id: int, label_size: float) -> None:
    ax.axis("off")
    ax.text(
        1.0,
        0.5,
        f"C{case_id}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=label_size,
        fontweight="bold",
    )


def draw_ring(ax, case: dict[str, object]) -> None:
    time_segments = np.asarray(case["time_unit_segments"], dtype=int)
    time_units = len(time_segments)
    if time_units == 0:
        ax.axis("off")
        return

    unit_counts = time_unit_item_counts(case, time_units)
    best_segment = None
    segments = active_sorted_segments(case)
    if segments:
        best_segment = int(segments[0]["segment_id"])

    theta = np.linspace(0.0, 2 * np.pi, time_units, endpoint=False)
    width = (2 * np.pi / time_units) * 0.995
    colors = []
    edge_colors = []
    line_widths = []
    for unit_idx, segment_id in enumerate(time_segments):
        if int(unit_counts[unit_idx]) == 0:
            colors.append("#e6e6e6")
            edge_colors.append("#ffffff")
            line_widths.append(0.35)
        elif best_segment is not None and int(segment_id) == best_segment:
            colors.append("#c1121f")
            edge_colors.append("#5c0007")
            line_widths.append(0.85)
        else:
            colors.append(SEGMENT_PALETTE[int(segment_id) % len(SEGMENT_PALETTE)])
            edge_colors.append("white")
            line_widths.append(0.45)

    ax.bar(
        theta,
        np.ones(time_units) * 0.34,
        width=width,
        bottom=0.56,
        color=colors,
        edgecolor=edge_colors,
        linewidth=line_widths,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
def similarity_grid(cases: list[dict[str, object]], max_segments: int | None):
    sorted_by_case = [active_sorted_segments(case) for case in cases]
    inferred_max = max(len(segments) for segments in sorted_by_case)
    column_count = inferred_max if max_segments is None else min(max_segments, inferred_max)
    similarities = np.full((len(cases), column_count), np.nan, dtype=float)
    annotations = np.full((len(cases), column_count), "", dtype=object)

    for row_idx, segments in enumerate(sorted_by_case):
        for col_idx, segment in enumerate(segments[:column_count]):
            similarities[row_idx, col_idx] = float(segment["similarity"])
            annotations[row_idx, col_idx] = f"S{segment['segment_id']}\n{float(segment['similarity']):.2f}"
    return similarities, annotations


def draw_heatmap(ax, cases: list[dict[str, object]], max_segments: int | None, title: str) -> None:
    similarities, annotations = similarity_grid(cases, max_segments)
    finite_values = similarities[np.isfinite(similarities)]
    color_min = min(0.0, float(finite_values.min()))
    color_max = max(0.1, float(finite_values.max()))
    image = ax.imshow(
        np.ma.masked_invalid(similarities),
        aspect="auto",
        cmap="Reds",
        vmin=color_min,
        vmax=color_max,
    )
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
    ax.set_xticks(range(similarities.shape[1]))
    ax.set_xticklabels([f"Top-{idx + 1}" for idx in range(similarities.shape[1])], fontsize=7.5)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([f"C{idx + 1}" for idx in range(len(cases))], fontsize=7.5)
    ax.tick_params(axis="both", length=2, pad=1)

    for row_idx in range(similarities.shape[0]):
        for col_idx in range(similarities.shape[1]):
            if not np.isfinite(similarities[row_idx, col_idx]):
                continue
            normalized = (similarities[row_idx, col_idx] - color_min) / max(color_max - color_min, 1e-8)
            ax.text(
                col_idx,
                row_idx,
                annotations[row_idx, col_idx],
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if normalized > 0.54 else "#1f1f1f",
                fontweight="bold" if col_idx == 0 else "normal",
            )
    return image


def plot_vertical(base_dir: Path, output_prefix: Path, max_cases: int, max_segments: int | None) -> None:
    datasets = [
        (dataset_name, display_name, time_name, load_dataset_cases(base_dir, dataset_name, max_cases))
        for dataset_name, display_name, time_name in DATASETS
    ]

    fig = plt.figure(figsize=(7.1, 6.2))
    outer = fig.add_gridspec(len(datasets), 1, hspace=0.38)
    for row_idx, (_, display_name, time_name, cases) in enumerate(datasets):
        panel = outer[row_idx].subgridspec(1, 2, width_ratios=[4.6, 1.4], wspace=0.15)
        heatmap_ax = fig.add_subplot(panel[0, 0])
        draw_heatmap(heatmap_ax, cases, max_segments, f"{display_name} ({time_name})")

        rings = panel[0, 1].subgridspec(len(cases), 2, width_ratios=[0.42, 1.0], wspace=0.06, hspace=0.02)
        for case_idx, case in enumerate(cases, start=1):
            label_ax = fig.add_subplot(rings[case_idx - 1, 0])
            draw_case_label(label_ax, case_idx, label_size=7.5)
            ring_ax = fig.add_subplot(rings[case_idx - 1, 1], projection="polar")
            draw_ring(ring_ax, case)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.975, bottom=0.055)
    fig.savefig(f"{output_prefix}_vertical.pdf")
    fig.savefig(f"{output_prefix}_vertical.png", dpi=300)
    plt.close(fig)


def plot_horizontal(base_dir: Path, output_prefix: Path, max_cases: int, max_segments: int | None) -> None:
    datasets = [
        (dataset_name, display_name, time_name, load_dataset_cases(base_dir, dataset_name, max_cases))
        for dataset_name, display_name, time_name in DATASETS
    ]

    fig = plt.figure(figsize=(7.1, 2.55))
    outer = fig.add_gridspec(1, len(datasets), wspace=0.18)
    for col_idx, (_, display_name, time_name, cases) in enumerate(datasets):
        panel = outer[col_idx].subgridspec(1, 2, width_ratios=[3.4, 1.08], wspace=0.1)
        heatmap_ax = fig.add_subplot(panel[0, 0])
        draw_heatmap(heatmap_ax, cases, max_segments, f"{display_name}\n({time_name})")

        rings = panel[0, 1].subgridspec(len(cases), 2, width_ratios=[0.45, 1.0], wspace=0.06, hspace=0.02)
        for case_idx, case in enumerate(cases, start=1):
            label_ax = fig.add_subplot(rings[case_idx - 1, 0])
            draw_case_label(label_ax, case_idx, label_size=5.6)
            ring_ax = fig.add_subplot(rings[case_idx - 1, 1], projection="polar")
            draw_ring(ring_ax, case)

    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.12)
    fig.savefig(f"{output_prefix}_horizontal.pdf")
    fig.savefig(f"{output_prefix}_horizontal.png", dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper-ready combined case-study figures.")
    parser.add_argument("--input-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_DIR / "case_study_paper"))
    parser.add_argument("--max-cases", default=5, type=int)
    parser.add_argument("--max-segments", default=None, type=int)
    parser.add_argument("--layout", default="both", choices=["vertical", "horizontal", "both"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(args.input_dir)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.layout in {"vertical", "both"}:
        plot_vertical(base_dir, output_prefix, args.max_cases, args.max_segments)
    if args.layout in {"horizontal", "both"}:
        plot_horizontal(base_dir, output_prefix, args.max_cases, args.max_segments)


if __name__ == "__main__":
    main()
