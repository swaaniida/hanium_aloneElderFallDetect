from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = {"raw": "#8B95A1", "baseline": "#2878B5", "dropout": "#F28E2B"}
METRICS = [
    ("accuracy", "정확도"),
    ("balanced_accuracy", "균형 정확도"),
    ("fall_precision", "낙상 정밀도"),
    ("fall_recall", "낙상 재현율"),
    ("fall_f1", "낙상 F1"),
    ("non_fall_specificity", "비낙상 특이도"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, default=Path("run_pilot100"))
    parser.add_argument(
        "--dropout-run", type=Path, default=Path("run_location1_seed42")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/location1_temporal_dropout_comparison"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def plot_one_training_row(axes, metrics: dict, title: str, color: str) -> None:
    head = metrics["head_history"]
    fine = metrics["finetune_history"]
    history = head + fine
    epochs = np.arange(1, len(history) + 1)
    boundary = len(head) + 0.5
    best_fine = int(np.argmin([row["validation_loss"] for row in fine]))
    best_epoch = len(head) + best_fine + 1

    for axis in axes:
        axis.axvspan(0.5, boundary, color="#DCEAF5", alpha=0.65)
        axis.axvspan(boundary, len(history) + 0.5, color="#FCE8D5", alpha=0.65)
        axis.axvline(boundary, color="#6B7280", linestyle="--", linewidth=1.2)

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        color=color,
        linewidth=2.2,
        label="Train",
    )
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        color="#D84A4A",
        linewidth=2.2,
        label="Validation",
    )
    axes[0].scatter(
        best_epoch,
        history[best_epoch - 1]["validation_loss"],
        marker="*",
        s=180,
        color="#18864B",
        zorder=5,
        label=f"선택 epoch {best_epoch}",
    )
    axes[0].set_ylabel(f"{title}\nBCE Loss", fontweight="bold")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper right", ncol=3)

    axes[1].plot(
        epochs,
        [row["train_accuracy"] * 100 for row in history],
        color=color,
        linewidth=2.2,
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] * 100 for row in history],
        color="#D84A4A",
        linewidth=2.2,
        label="Validation",
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(35, 103)
    axes[1].legend(loc="lower right", ncol=2)


def plot_training(baseline: dict, dropout: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
    figure.suptitle(
        "위치 1 TSSTG 학습 과정 — Temporal Pose Dropout 적용 전·후",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    plot_one_training_row(
        axes[0], baseline, "기존 증강\n(dropout 없음)", COLORS["baseline"]
    )
    plot_one_training_row(
        axes[1], dropout, "Temporal dropout\n(p=0.35, 1–4 frames)", COLORS["dropout"]
    )
    axes[0, 0].set_title("손실 곡선", fontweight="bold")
    axes[0, 1].set_title("정확도 곡선", fontweight="bold")
    for axis in axes[1]:
        axis.set_xlabel("Epoch  |  1–15: Head 학습, 이후: 전체 Fine-tuning")
    figure.text(
        0.5,
        0.018,
        "파란 배경: Head 학습   ·   주황 배경: 전체 Fine-tuning   ·   초록 별: 최종 선택 checkpoint",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.955))
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_confusion(axis, matrix: list[list[int]], title: str, subtitle: str) -> None:
    array = np.asarray(matrix)
    axis.imshow(array, cmap="Blues", vmin=0, vmax=10)
    for row in range(2):
        for column in range(2):
            value = int(array[row, column])
            axis.text(
                column,
                row,
                value,
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
                color="white" if value >= 6 else "#17202A",
            )
    axis.set_xticks([0, 1], ["NON_FALL", "FALL"])
    axis.set_yticks([0, 1], ["NON_FALL", "FALL"])
    axis.set_xlabel("예측")
    axis.set_ylabel("정답")
    axis.set_title(f"{title}\n{subtitle}", fontweight="bold", pad=12)


def plot_performance(baseline: dict, dropout: dict, output: Path) -> None:
    raw = baseline["original_7class_tsstg"]
    old = baseline["fine_tuned_binary_tsstg"]
    new = dropout["fine_tuned_binary_tsstg"]
    x = np.arange(len(METRICS))
    width = 0.25

    figure = plt.figure(figsize=(17, 11))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.42)
    axis = figure.add_subplot(grid[0, :])
    series = [
        ("Raw TSSTG", raw, COLORS["raw"]),
        ("기존 위치 1 모델", old, COLORS["baseline"]),
        ("Temporal dropout 모델", new, COLORS["dropout"]),
    ]
    for index, (name, metric, color) in enumerate(series):
        values = [metric[key] * 100 for key, _ in METRICS]
        bars = axis.bar(x + (index - 1) * width, values, width, label=name, color=color)
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=10, fontweight="bold")
    axis.set_xticks(x, [label for _, label in METRICS])
    axis.set_ylim(0, 113)
    axis.set_ylabel("점수 (%)")
    axis.set_title(
        "동일한 공통 Test 20개 성능 비교 (FALL 10 / NON_FALL 10)",
        fontsize=19,
        fontweight="bold",
        pad=14,
    )
    axis.legend(loc="lower left", ncol=3, frameon=True)

    draw_confusion(
        figure.add_subplot(grid[1, 0]),
        raw["confusion_matrix"],
        "Raw TSSTG",
        f"Accuracy {raw['accuracy'] * 100:.0f}%",
    )
    draw_confusion(
        figure.add_subplot(grid[1, 1]),
        old["confusion_matrix"],
        "기존 위치 1 모델",
        f"Accuracy {old['accuracy'] * 100:.0f}%",
    )
    draw_confusion(
        figure.add_subplot(grid[1, 2]),
        new["confusion_matrix"],
        "Temporal dropout 모델",
        f"Accuracy {new['accuracy'] * 100:.0f}%",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_location_breakdown(baseline: dict, dropout: dict, output: Path) -> None:
    scopes = [
        ("공통 Test 20", None),
        ("위치 1 Test 10", "location_1"),
        ("위치 2 Test 10", "location_2"),
    ]
    keys = ["accuracy", "fall_recall", "fall_f1", "non_fall_specificity"]
    labels = ["정확도", "낙상 재현율", "낙상 F1", "비낙상 특이도"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.3), sharey=True)
    x = np.arange(len(keys))
    width = 0.34
    for axis, (title, location) in zip(axes, scopes):
        if location is None:
            old = baseline["fine_tuned_binary_tsstg"]
            new = dropout["fine_tuned_binary_tsstg"]
        else:
            old = baseline["metrics_by_location"][location][
                "fine_tuned_binary_tsstg"
            ]
            new = dropout["metrics_by_location"][location][
                "fine_tuned_binary_tsstg"
            ]
        old_bars = axis.bar(
            x - width / 2,
            [old[key] * 100 for key in keys],
            width,
            color=COLORS["baseline"],
            label="기존",
        )
        new_bars = axis.bar(
            x + width / 2,
            [new[key] * 100 for key in keys],
            width,
            color=COLORS["dropout"],
            label="Dropout",
        )
        axis.bar_label(old_bars, fmt="%.0f", padding=3, fontsize=10)
        axis.bar_label(new_bars, fmt="%.0f", padding=3, fontsize=10)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(title, fontweight="bold")
        axis.set_ylim(0, 112)
    axes[0].set_ylabel("점수 (%)")
    axes[0].legend(loc="lower left", ncol=2)
    figure.suptitle(
        "장소별 일반화 비교 — 위치 1 학습 모델",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    figure.text(
        0.5,
        -0.02,
        "위치 1 성능은 동일 · 위치 2에서 dropout 모델의 NON_FALL 오탐 1건 증가",
        ha="center",
        fontsize=12,
        color="#B03A2E",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_summary_csv(baseline: dict, dropout: dict, output: Path) -> None:
    rows = []
    for scope, location in [
        ("common_test_20", None),
        ("location_1_test_10", "location_1"),
        ("location_2_test_10", "location_2"),
    ]:
        for model, report, model_key in [
            ("raw_tsstg", baseline, "original_7class_tsstg"),
            ("baseline", baseline, "fine_tuned_binary_tsstg"),
            ("temporal_dropout", dropout, "fine_tuned_binary_tsstg"),
        ]:
            metrics = report[model_key] if location is None else report[
                "metrics_by_location"
            ][location][model_key]
            rows.append(
                {"scope": scope, "model": model, **{key: metrics[key] for key, _ in METRICS}}
            )
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    baseline_training = load_json(args.baseline_run / "metrics.json")
    dropout_training = load_json(args.dropout_run / "metrics.json")
    baseline_comparison = load_json(
        args.baseline_run / "common_test_comparison" / "comparison_metrics.json"
    )
    dropout_comparison = load_json(
        args.dropout_run / "common_test_comparison" / "comparison_metrics.json"
    )
    plot_training(
        baseline_training,
        dropout_training,
        args.output / "01_training_curves.png",
    )
    plot_performance(
        baseline_comparison,
        dropout_comparison,
        args.output / "02_common_test_performance.png",
    )
    plot_location_breakdown(
        baseline_comparison,
        dropout_comparison,
        args.output / "03_location_generalization.png",
    )
    write_summary_csv(
        baseline_comparison,
        dropout_comparison,
        args.output / "metrics_comparison.csv",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
