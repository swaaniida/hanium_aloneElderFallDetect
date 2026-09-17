from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced\naccuracy"),
    ("fall_precision", "Fall\nprecision"),
    ("fall_recall", "Fall\nrecall"),
    ("fall_f1", "Fall F1"),
    ("non_fall_specificity", "Non-fall\nspecificity"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot location-1 pilot training and model-comparison results"
    )
    parser.add_argument(
        "--training-metrics", type=Path, default=Path("run_pilot100/metrics.json")
    )
    parser.add_argument(
        "--comparison-metrics",
        type=Path,
        default=Path("run_pilot100/model_comparison/comparison_metrics.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("run_pilot100/analysis")
    )
    parser.add_argument(
        "--comparison-filename",
        default="location1_model_comparison.png",
        help="Filename for the comparison figure",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def plot_training(metrics: dict, output: Path) -> None:
    head = metrics["head_history"]
    finetune = metrics["finetune_history"]
    history = head + finetune
    epochs = np.arange(1, len(history) + 1)
    phase_boundary = len(head) + 0.5
    best_local_index = int(
        np.argmin([row["validation_loss"] for row in finetune])
    )
    best_epoch = len(head) + best_local_index + 1
    best_loss = finetune[best_local_index]["validation_loss"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    figure.suptitle(
        "Location 1 pilot training (80 train / 10 validation)",
        fontsize=15,
        fontweight="bold",
    )

    for axis in axes:
        axis.axvspan(0.5, phase_boundary, color="#4C78A8", alpha=0.07)
        axis.axvspan(
            phase_boundary, len(history) + 0.5, color="#F58518", alpha=0.07
        )
        axis.axvline(phase_boundary, color="#555555", linestyle="--", linewidth=1)
        axis.text(7.5, 0.96, "Head training", transform=axis.get_xaxis_transform(),
                  ha="center", va="top", color="#315B7D")
        axis.text(
            len(head) + len(finetune) / 2,
            0.96,
            "Full fine-tuning",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#A5540A",
        )

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        marker="o",
        markersize=3,
        label="Train loss",
        color="#4C78A8",
    )
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        marker="o",
        markersize=3,
        label="Validation loss",
        color="#E45756",
    )
    axes[0].scatter(
        [best_epoch], [best_loss], marker="*", s=180, color="#54A24B", zorder=5,
        label=f"Selected checkpoint (epoch {best_epoch})"
    )
    axes[0].annotate(
        f"best val loss={best_loss:.4f}",
        (best_epoch, best_loss),
        xytext=(best_epoch - 9, 0.16),
        arrowprops={"arrowstyle": "->", "color": "#54A24B"},
        fontsize=9,
    )
    axes[0].set_ylabel("Binary cross-entropy loss")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="center right", ncol=1)

    axes[1].plot(
        epochs,
        [row["train_accuracy"] * 100 for row in history],
        marker="o",
        markersize=3,
        label="Train accuracy",
        color="#4C78A8",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] * 100 for row in history],
        marker="o",
        markersize=3,
        label="Validation accuracy",
        color="#E45756",
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Epoch (head 1-15, fine-tuning 16-38)")
    axes[1].set_ylim(40, 103)
    axes[1].legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def draw_confusion(axis, matrix: np.ndarray, title: str) -> None:
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(5, int(matrix.max())))
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="white" if value >= 3 else "#222222",
            )
    axis.set_xticks([0, 1], ["NON_FALL", "FALL"])
    axis.set_yticks([0, 1], ["NON_FALL", "FALL"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title, fontweight="bold")
    return image


def plot_comparison(comparison: dict, output: Path) -> None:
    original = comparison["original_7class_tsstg"]
    binary = comparison["fine_tuned_binary_tsstg"]
    x = np.arange(len(METRIC_LABELS))
    width = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(14, 8))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.25, 1])
    bars_axis = figure.add_subplot(grid[0, :])
    original_values = [original[key] * 100 for key, _ in METRIC_LABELS]
    binary_values = [binary[key] * 100 for key, _ in METRIC_LABELS]
    original_confusion = np.asarray(original["confusion_matrix"])
    binary_confusion = np.asarray(binary["confusion_matrix"])
    non_fall_count = int(binary_confusion[0].sum())
    fall_count = int(binary_confusion[1].sum())
    original_bars = bars_axis.bar(
        x - width / 2,
        original_values,
        width,
        label="Original 7-class TSSTG (mapped)",
        color="#9C9C9C",
    )
    binary_bars = bars_axis.bar(
        x + width / 2,
        binary_values,
        width,
        label="Fine-tuned binary TSSTG",
        color="#F58518",
    )
    bars_axis.bar_label(original_bars, fmt="%.0f%%", padding=3, fontsize=9)
    bars_axis.bar_label(binary_bars, fmt="%.0f%%", padding=3, fontsize=9)
    bars_axis.set_xticks(x, [label for _, label in METRIC_LABELS])
    bars_axis.set_ylim(0, 112)
    bars_axis.set_ylabel("Score (%)")
    bars_axis.set_title(
        f"Location 1 test comparison ({fall_count} fall / {non_fall_count} non-fall)",
        fontsize=15,
        fontweight="bold",
    )
    bars_axis.legend(loc="lower left", ncol=2)

    original_axis = figure.add_subplot(grid[1, 0])
    binary_axis = figure.add_subplot(grid[1, 1])
    draw_confusion(
        original_axis,
        original_confusion,
        f"Original mapped TSSTG — accuracy {original['accuracy'] * 100:.0f}%",
    )
    draw_confusion(
        binary_axis,
        binary_confusion,
        f"Fine-tuned binary TSSTG — accuracy {binary['accuracy'] * 100:.0f}%",
    )

    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training = load_json(args.training_metrics)
    comparison = load_json(args.comparison_metrics)
    training_path = args.output / "location1_training_curves.png"
    comparison_path = args.output / args.comparison_filename
    plot_training(training, training_path)
    plot_comparison(comparison, comparison_path)
    print(training_path.resolve())
    print(comparison_path.resolve())


if __name__ == "__main__":
    main()
