from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = {
    "raw": "#98A1AE",
    "baseline": "#31688E",
    "location1": "#35B779",
    "location2": "#F4A261",
    "full200": "#8E5BE8",
}
METRICS = [
    ("accuracy", "정확도"),
    ("balanced_accuracy", "균형 정확도"),
    ("fall_precision", "낙상 정밀도\n(낙상 예측 중 실제 낙상)"),
    ("fall_recall", "낙상 재현율\n(실제 낙상 중 낙상으로 감지)"),
    ("fall_f1", "낙상 F1"),
    ("non_fall_specificity", "비낙상 특이도\n(실제 비낙상 중 비낙상으로 판정)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/seed42_all_model_comparison")
    )
    parser.add_argument(
        "--comparison-subdir", default="common_test_comparison",
        help="Comparison result directory inside each run",
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
            "legend.fontsize": 10,
        }
    )


def plot_training_row(axes, metrics: dict, name: str, detail: str, color: str) -> None:
    head = metrics["head_history"]
    fine = metrics["finetune_history"]
    history = head + fine
    epochs = np.arange(1, len(history) + 1)
    boundary = len(head) + 0.5
    best_index = int(np.argmin([row["validation_loss"] for row in fine]))
    best_epoch = len(head) + best_index + 1
    for axis in axes:
        axis.axvspan(0.5, boundary, color="#DDECF5", alpha=0.65)
        axis.axvspan(boundary, len(history) + 0.5, color="#FCE9D7", alpha=0.65)
        axis.axvline(boundary, color="#697386", linestyle="--", linewidth=1.1)
    axes[0].plot(
        epochs, [row["train_loss"] for row in history], color=color,
        linewidth=2.1, label="Train"
    )
    axes[0].plot(
        epochs, [row["validation_loss"] for row in history], color="#D64545",
        linewidth=2.1, label="Validation"
    )
    axes[0].scatter(
        best_epoch, history[best_epoch - 1]["validation_loss"], marker="*",
        s=165, color="#137C4A", zorder=5, label=f"선택 epoch {best_epoch}"
    )
    axes[0].set_ylabel("BCE Loss")
    axes[0].text(
        0.02, 0.90, f"{name}  |  {detail}", transform=axes[0].transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": color},
    )
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].plot(
        epochs, [row["train_accuracy"] * 100 for row in history], color=color,
        linewidth=2.1, label="Train"
    )
    axes[1].plot(
        epochs, [row["validation_accuracy"] * 100 for row in history],
        color="#D64545", linewidth=2.1, label="Validation"
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(35, 103)
    axes[1].legend(loc="lower right", ncol=2)


def plot_training_suite(
    training: dict[str, dict],
    output: Path,
    title: str = "Seed 42 신규 TSSTG 학습 과정 — Temporal Pose Dropout 적용",
) -> None:
    specs = [
        ("location1", "위치 1 모델", "Train 80", COLORS["location1"]),
        ("location2", "위치 2 모델", "Train 79 · E101 제외", COLORS["location2"]),
        ("full200", "전체 결합 모델", "Train 159 · Val 20 · Test 20", COLORS["full200"]),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(17, 14), sharex="col")
    figure.suptitle(
        title,
        fontsize=21, fontweight="bold", y=0.99
    )
    for row, (key, name, detail, color) in enumerate(specs):
        plot_training_row(axes[row], training[key], name, detail, color)
    axes[0, 0].set_title("손실 곡선", fontweight="bold")
    axes[0, 1].set_title("정확도 곡선", fontweight="bold")
    for axis in axes[-1]:
        axis.set_xlabel("Epoch  |  파란 배경: Head 학습 · 주황 배경: 전체 Fine-tuning")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_confusion(axis, matrix: list[list[int]], title: str, accuracy: float) -> None:
    array = np.asarray(matrix)
    axis.imshow(array, cmap="Blues", vmin=0, vmax=10)
    for row in range(2):
        for column in range(2):
            value = int(array[row, column])
            axis.text(
                column, row, value, ha="center", va="center", fontsize=21,
                fontweight="bold", color="white" if value >= 6 else "#17202A"
            )
    axis.set_xticks([0, 1], ["비낙상", "낙상"])
    axis.set_yticks([0, 1], ["비낙상", "낙상"])
    axis.set_xlabel("예측")
    axis.set_ylabel("")
    axis.text(
        -0.12,
        1.06,
        "정답",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )
    axis.set_title(f"{title}\n정확도 {accuracy * 100:.0f}%", fontweight="bold")


def plot_common_test(
    models: list[tuple[str, dict, str]], output: Path, test_samples: int
) -> None:
    figure = plt.figure(figsize=(21, 12))
    grid = figure.add_gridspec(
        2,
        5,
        height_ratios=[1.25, 1],
        hspace=0.64,
        wspace=0.58,
        left=0.055,
        right=0.985,
    )
    axis = figure.add_subplot(grid[0, :])
    x = np.arange(len(METRICS))
    width = 0.16
    offsets = np.arange(len(models)) - (len(models) - 1) / 2
    for offset, (name, metric, color) in zip(offsets, models):
        values = [metric[key] * 100 for key, _ in METRICS]
        bars = axis.bar(x + offset * width, values, width, label=name, color=color)
        axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=8.5, fontweight="bold")
    axis.set_xticks(x, [label for _, label in METRICS])
    axis.set_ylim(0, 114)
    axis.set_ylabel("점수 (%)")
    axis.set_title(
        f"동일한 Test {test_samples}개 모델 성능 비교 "
        f"(낙상 {test_samples // 2} / 비낙상 {test_samples // 2})",
        fontsize=20, fontweight="bold", pad=14
    )
    axis.legend(loc="lower left", ncol=5, frameon=True)
    for column, (name, metric, _) in enumerate(models):
        draw_confusion(
            figure.add_subplot(grid[1, column]), metric["confusion_matrix"],
            name, metric["accuracy"]
        )
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_location_heatmaps(
    model_reports: list[tuple[str, dict]], output: Path, samples_per_location: int
) -> None:
    metric_specs = [
        ("accuracy", "정확도"),
        ("fall_recall", "낙상 재현율\n(실제 낙상 중 낙상으로 감지)"),
        ("fall_f1", "낙상 F1"),
        (
            "non_fall_specificity",
            "비낙상 특이도\n(실제 비낙상 중 비낙상으로 판정)",
        ),
    ]
    names = [name for name, _ in model_reports]
    locations = ["location_1", "location_2"]
    figure, axes = plt.subplots(1, 4, figsize=(19, 7.2))
    for axis, (metric_key, title) in zip(axes, metric_specs):
        values = np.asarray(
            [
                [
                    report["metrics_by_location"][location]["fine_tuned_binary_tsstg"][metric_key] * 100
                    for location in locations
                ]
                for _, report in model_reports
            ]
        )
        axis.imshow(values, cmap="YlGnBu", vmin=40, vmax=100, aspect="auto")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                axis.text(
                    column, row, f"{value:.0f}", ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if value >= 80 else "#17202A"
                )
        axis.set_xticks(
            [0, 1],
            [
                f"위치 1\nTest {samples_per_location}",
                f"위치 2\nTest {samples_per_location}",
            ],
        )
        axis.set_yticks(np.arange(len(names)), names)
        axis.set_title(title, fontweight="bold", pad=12)
    figure.suptitle(
        "장소별 일반화 성능 히트맵", fontsize=21, fontweight="bold", y=1.02
    )
    figure.text(
        0.5, 0.01,
        "전체 결합 모델이 두 장소에서 가장 높은 종합 성능을 유지",
        ha="center", fontsize=12.5, color="#A12B3A", fontweight="bold"
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.98))
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_csv(models: list[tuple[str, dict, str]], output: Path) -> None:
    rows = []
    for name, metric, _ in models:
        rows.append(
            {"model": name, "scope": "common_test_20", **{key: metric[key] for key, _ in METRICS}}
        )
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    run_paths = {
        "location1": Path("run_location1_seed42"),
        "location2": Path("run_location2_seed42"),
        "full200": Path("run_full200_seed42"),
    }
    training = {key: load_json(path / "metrics.json") for key, path in run_paths.items()}
    reports = {
        key: load_json(path / args.comparison_subdir / "comparison_metrics.json")
        for key, path in run_paths.items()
    }
    baseline = load_json(
        Path("run_pilot100") / args.comparison_subdir / "comparison_metrics.json"
    )
    models = [
        ("Raw TSSTG", baseline["original_7class_tsstg"], COLORS["raw"]),
        ("위치 1 기존", baseline["fine_tuned_binary_tsstg"], COLORS["baseline"]),
        ("위치 1 Dropout", reports["location1"]["fine_tuned_binary_tsstg"], COLORS["location1"]),
        ("위치 2 Dropout", reports["location2"]["fine_tuned_binary_tsstg"], COLORS["location2"]),
        ("전체 결합 Dropout", reports["full200"]["fine_tuned_binary_tsstg"], COLORS["full200"]),
    ]
    plot_training_suite(training, args.output / "01_new_models_training_curves.png")
    test_samples = int(reports["full200"]["samples"])
    samples_per_location = int(
        reports["full200"]["metrics_by_location"]["location_1"]["samples"]
    )
    plot_common_test(
        models, args.output / "02_all_models_common_test.png", test_samples
    )
    plot_location_heatmaps(
        [
            ("위치 1 기존", baseline),
            ("위치 1 Dropout", reports["location1"]),
            ("위치 2 Dropout", reports["location2"]),
            ("전체 결합 Dropout", reports["full200"]),
        ],
        args.output / "03_location_generalization_heatmap.png",
        samples_per_location,
    )
    write_csv(models, args.output / "common_test_metrics.csv")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
