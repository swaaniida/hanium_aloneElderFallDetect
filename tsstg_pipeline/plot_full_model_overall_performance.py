from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .plot_seed42_model_suite import COLORS, set_style


REPORT = Path(
    "run_full200_nodropout_seed42/test40_comparison/comparison_metrics.json"
)
OUTPUT = Path("analysis/seed42_no_temporal_dropout/own_location")
METRICS = [
    ("accuracy", "정확도"),
    ("balanced_accuracy", "균형\n정확도"),
    ("fall_precision", "낙상\n정밀도"),
    ("fall_recall", "낙상\n재현율"),
    ("fall_f1", "낙상 F1"),
    ("non_fall_specificity", "비낙상\n특이도"),
]


def load_report() -> dict:
    with REPORT.open(encoding="utf-8") as stream:
        return json.load(stream)


def plot_bar(report: dict, output: Path) -> None:
    raw = report["original_7class_tsstg"]
    model = report["fine_tuned_binary_tsstg"]
    x = np.arange(len(METRICS))
    width = 0.34

    figure, axis = plt.subplots(figsize=(12, 5.8))
    raw_bars = axis.bar(
        x - width / 2,
        [raw[key] * 100 for key, _ in METRICS],
        width,
        color=COLORS["raw"],
        label="기존 TSSTG",
    )
    model_bars = axis.bar(
        x + width / 2,
        [model[key] * 100 for key, _ in METRICS],
        width,
        color=COLORS["full200"],
        label="전체 결합 모델",
    )
    axis.bar_label(raw_bars, fmt="%.0f", padding=3, fontsize=13)
    axis.bar_label(
        model_bars,
        fmt="%.0f",
        padding=3,
        fontsize=13,
        fontweight="bold",
    )
    axis.set_xticks(x, [label for _, label in METRICS], fontsize=13)
    axis.set_ylim(0, 112)
    axis.set_ylabel("점수 (%)", fontsize=13)
    axis.set_title(
        "전체 결합 모델 성능 · 두 위치 Test 40",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    axis.legend(loc="lower left", ncol=2, fontsize=12, frameon=True)
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_confusion(report: dict, output: Path) -> None:
    matrix = np.asarray(
        report["fine_tuned_binary_tsstg"]["confusion_matrix"], dtype=int
    )
    figure, axis = plt.subplots(figsize=(6.8, 6.4))
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=36,
                fontweight="bold",
                color="white" if matrix[row, column] > threshold else "#17202A",
            )
    axis.set_xticks([0, 1], ["비낙상", "낙상"], fontsize=16)
    axis.set_yticks([0, 1], ["비낙상", "낙상"], fontsize=16)
    axis.set_xlabel("예측", fontsize=16, labelpad=8)
    axis.set_ylabel("정답", fontsize=16, rotation=0, labelpad=28)
    axis.set_title(
        "전체 결합 모델 · 정확도 85%",
        fontsize=21,
        fontweight="bold",
        pad=18,
    )
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    set_style()
    report = load_report()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_bar(report, OUTPUT / "02_overall_performance_bar.png")
    plot_confusion(report, OUTPUT / "02_overall_confusion_heatmap.png")
    print((OUTPUT / "02_overall_performance_bar.png").resolve())
    print((OUTPUT / "02_overall_confusion_heatmap.png").resolve())


if __name__ == "__main__":
    main()
