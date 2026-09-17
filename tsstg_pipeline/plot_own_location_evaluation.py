from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    ConnectionPatch,
    FancyArrowPatch,
    FancyBboxPatch,
)

from .plot_seed42_model_suite import COLORS, draw_confusion, load_json, set_style


OUTPUT = Path("analysis/seed42_no_temporal_dropout/own_location")
METRICS = [
    ("accuracy", "정확도"),
    ("balanced_accuracy", "균형\n정확도"),
    ("fall_precision", "낙상 정밀도\n(예측 중 실제 낙상)"),
    ("fall_recall", "낙상 재현율\n(실제 낙상 감지)"),
    ("fall_f1", "낙상 F1"),
    ("non_fall_specificity", "비낙상 특이도\n(비낙상 정상 판정)"),
]


def load_reports() -> list[dict]:
    specs = [
        (
            "위치 1 모델",
            "자기 위치 Test 20",
            Path("run_location1_nodropout_seed42/own_location_test20_comparison/comparison_metrics.json"),
            COLORS["location1"],
        ),
        (
            "위치 2 모델",
            "자기 위치 Test 20",
            Path("run_location2_nodropout_seed42/own_location_test20_comparison/comparison_metrics.json"),
            COLORS["location2"],
        ),
        (
            "전체 결합 모델",
            "두 위치 Test 40",
            Path("run_full200_nodropout_seed42/test40_comparison/comparison_metrics.json"),
            COLORS["full200"],
        ),
    ]
    return [
        {
            "name": name,
            "scope": scope,
            "report": load_json(path),
            "color": color,
        }
        for name, scope, path, color in specs
    ]


def plot_panels(items: list[dict], output: Path) -> None:
    figure = plt.figure(figsize=(22, 12.5))
    grid = figure.add_gridspec(
        2, 3, height_ratios=[1.18, 1], hspace=0.52, wspace=0.22,
        left=0.045, right=0.985, top=0.90, bottom=0.08
    )
    figure.suptitle(
        "각 모델을 자기 데이터 범위에서 평가한 성능",
        fontsize=22,
        fontweight="bold",
        y=0.975,
    )
    x = np.arange(len(METRICS))
    width = 0.36
    for column, item in enumerate(items):
        report = item["report"]
        raw = report["original_7class_tsstg"]
        model = report["fine_tuned_binary_tsstg"]
        axis = figure.add_subplot(grid[0, column])
        raw_bars = axis.bar(
            x - width / 2,
            [raw[key] * 100 for key, _ in METRICS],
            width,
            color=COLORS["raw"],
            label="Raw TSSTG",
        )
        model_bars = axis.bar(
            x + width / 2,
            [model[key] * 100 for key, _ in METRICS],
            width,
            color=item["color"],
            label=item["name"],
        )
        axis.bar_label(raw_bars, fmt="%.0f", padding=2, fontsize=9)
        axis.bar_label(model_bars, fmt="%.0f", padding=2, fontsize=9, fontweight="bold")
        axis.set_xticks(x, [label for _, label in METRICS], fontsize=9)
        axis.set_ylim(0, 114)
        axis.set_title(
            f"{item['name']}  |  {item['scope']}",
            fontsize=16,
            fontweight="bold",
            pad=13,
        )
        if column == 0:
            axis.set_ylabel("점수 (%)")
        axis.legend(loc="lower left", ncol=2, fontsize=9)

        inner = grid[1, column].subgridspec(1, 2, wspace=0.48)
        draw_confusion(
            figure.add_subplot(inner[0, 0]),
            raw["confusion_matrix"],
            "Raw TSSTG",
            raw["accuracy"],
        )
        draw_confusion(
            figure.add_subplot(inner[0, 1]),
            model["confusion_matrix"],
            item["name"],
            model["accuracy"],
        )
    figure.text(
        0.5,
        0.025,
        "위치별 모델은 자기 위치 20개 · 전체 결합 모델은 두 위치 40개에서 평가",
        ha="center",
        fontsize=12,
        color="#4B5563",
        fontweight="bold",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_summary_heatmap(items: list[dict], output: Path) -> None:
    values = np.asarray(
        [
            [item["report"]["fine_tuned_binary_tsstg"][key] * 100 for key, _ in METRICS]
            for item in items
        ]
    )
    figure, axis = plt.subplots(figsize=(15.5, 6.5))
    axis.imshow(values, cmap="YlGnBu", vmin=30, vmax=100, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="white" if value >= 80 else "#17202A",
            )
    axis.set_xticks(np.arange(len(METRICS)), [label for _, label in METRICS])
    axis.set_yticks(
        np.arange(len(items)),
        [f"{item['name']}\n{item['scope']}" for item in items],
    )
    axis.set_title(
        "자기 평가 범위 기준 핵심 성능 요약",
        fontsize=21,
        fontweight="bold",
        pad=18,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_annotated_story(items: list[dict], output: Path) -> None:
    """Publication-style annotated duplicate of the own-scope comparison."""
    figure = plt.figure(figsize=(22, 16))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 0.62, 1],
        hspace=0.34,
        wspace=0.22,
        left=0.045,
        right=0.985,
        top=0.875,
        bottom=0.065,
    )
    figure.suptitle(
        "환경별 모델 성능과 전체 결합 모델의 일반화",
        fontsize=23,
        fontweight="bold",
        y=0.978,
    )
    story_metrics = [
        ("accuracy", "정확도"),
        ("balanced_accuracy", "균형\n정확도"),
        ("fall_precision", "낙상\n정밀도"),
        ("fall_recall", "낙상\n재현율"),
        ("fall_f1", "낙상 F1"),
        ("non_fall_specificity", "비낙상\n특이도"),
    ]
    x = np.arange(len(story_metrics))
    width = 0.36
    contexts = [
        "강의실 환경",
        "독거노인이 생활할 수 있는 가정 환경\n침대 낙상 포함",
        "강의실 + 가정 환경",
    ]
    bottom_pairs = []
    bar_axes = []

    for column, (item, context) in enumerate(zip(items, contexts)):
        report = item["report"]
        raw = report["original_7class_tsstg"]
        model = report["fine_tuned_binary_tsstg"]
        axis = figure.add_subplot(grid[0, column])
        bar_axes.append(axis)
        raw_bars = axis.bar(
            x - width / 2,
            [raw[key] * 100 for key, _ in story_metrics],
            width,
            color=COLORS["raw"],
            label="Raw TSSTG",
        )
        model_bars = axis.bar(
            x + width / 2,
            [model[key] * 100 for key, _ in story_metrics],
            width,
            color=item["color"],
            label=item["name"],
        )
        axis.bar_label(raw_bars, fmt="%.0f", padding=2, fontsize=9)
        axis.bar_label(model_bars, fmt="%.0f", padding=2, fontsize=9, fontweight="bold")
        axis.set_xticks(x, [label for _, label in story_metrics], fontsize=9.5)
        axis.set_ylim(0, 114)
        if column < 2:
            axis.set_title(
                f"{item['name']}  |  {item['scope']}\n{context}",
                fontsize=15,
                fontweight="bold",
                pad=14,
            )
        else:
            axis.set_title("")
            axis.text(
                0.5,
                1.18,
                "전체 결합 모델",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                color="#6335A5",
                bbox={
                    "boxstyle": "round,pad=0.28,rounding_size=0.45",
                    "facecolor": "#EEE6FF",
                    "edgecolor": "#9A78D4",
                    "linewidth": 1.4,
                },
            )
            axis.text(
                0.5,
                1.105,
                "위치 1 + 위치 2 Test 40",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=12.5,
                color="#303943",
                fontweight="bold",
            )
            axis.text(
                0.5,
                1.045,
                context,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=11.5,
                color="#59636D",
            )
        if column == 0:
            axis.set_ylabel("점수 (%)")
        axis.legend(loc="lower left", ncol=2, fontsize=9)

        inner = grid[2, column].subgridspec(1, 2, wspace=0.48)
        raw_axis = figure.add_subplot(inner[0, 0])
        model_axis = figure.add_subplot(inner[0, 1])
        draw_confusion(
            raw_axis,
            raw["confusion_matrix"],
            "Raw TSSTG",
            raw["accuracy"],
        )
        draw_confusion(
            model_axis,
            model["confusion_matrix"],
            item["name"],
            model["accuracy"],
        )
        if column == 2:
            model_axis.set_title("")
            model_axis.text(
                0.5,
                1.17,
                "전체 결합 모델",
                transform=model_axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                color="#6335A5",
                bbox={
                    "boxstyle": "round,pad=0.25,rounding_size=0.4",
                    "facecolor": "#EEE6FF",
                    "edgecolor": "#9A78D4",
                    "linewidth": 1.25,
                },
                clip_on=False,
            )
            model_axis.text(
                0.5,
                1.075,
                f"정확도 {model['accuracy'] * 100:.0f}%",
                transform=model_axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color="#303943",
                clip_on=False,
            )
        gain = (model["accuracy"] - raw["accuracy"]) * 100
        arrow = ConnectionPatch(
            xyA=(0.62, 1.34),
            coordsA=raw_axis.transAxes,
            xyB=(0.38, 1.34),
            coordsB=model_axis.transAxes,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.28",
            mutation_scale=19,
            linewidth=2.2,
            color=item["color"],
            clip_on=False,
        )
        figure.add_artist(arrow)
        bottom_pairs.append((raw_axis, model_axis, gain, item["color"]))

    figure.canvas.draw()
    bar_bottom = min(axis.get_position().y0 for axis in bar_axes)
    figure.text(
        0.5,
        bar_bottom - 0.030,
        "정밀도: 낙상 예측 중 실제 낙상   ·   재현율: 실제 낙상 중 감지 성공   ·   특이도: 실제 비낙상 중 정상 판정",
        ha="center",
        va="center",
        fontsize=11.2,
        color="#505B66",
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.25",
            "facecolor": "#F5F7F9",
            "edgecolor": "#D7DDE3",
            "linewidth": 1.0,
        },
    )

    flow = figure.add_subplot(grid[1, :])
    flow.set_axis_off()
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)

    def flow_card(x0, y0, width_box, height_box, title, lines, color, facecolor):
        box = FancyBboxPatch(
            (x0, y0),
            width_box,
            height_box,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.7,
            edgecolor=color,
            facecolor=facecolor,
        )
        box.set_path_effects(
            [pe.SimplePatchShadow(offset=(3, -3), alpha=0.16), pe.Normal()]
        )
        flow.add_patch(box)
        flow.text(
            x0 + width_box / 2,
            y0 + height_box - 0.070,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        flow.plot(
            [x0 + 0.055, x0 + width_box - 0.055],
            [y0 + height_box - 0.135, y0 + height_box - 0.135],
            color=color,
            linewidth=2.7,
            solid_capstyle="round",
        )
        for index, (line, line_color, bold) in enumerate(lines):
            flow.text(
                x0 + 0.028,
                y0 + height_box - 0.205 - index * 0.077,
                line,
                ha="left",
                va="center",
                fontsize=10.6,
                color=line_color,
                fontweight="bold" if bold else "normal",
            )

    flow_card(
        0.035,
        0.57,
        0.285,
        0.34,
        "위치 1 모델",
        [
            ("강의실 환경", "#157A52", True),
            ("정형화된 배경과 낙상 동선", "#44515D", False),
        ],
        COLORS["location1"],
        "#F4FBF7",
    )
    flow_card(
        0.035,
        0.075,
        0.285,
        0.40,
        "위치 2 모델",
        [
            ("독거노인이 생활할 수 있는 가정 환경", "#9A4C13", True),
            ("침대에서 떨어지는 낙상 포함", "#9A4C13", True),
            ("침대 높이·이불 가림·큰 자세 변화 → 난도 상승", "#59636D", False),
        ],
        COLORS["location2"],
        "#FFF8F1",
    )
    flow_card(
        0.665,
        0.24,
        0.30,
        0.53,
        "전체 결합 모델",
        [
            ("위치 1 + 위치 2를 함께 학습", "#6840A8", True),
            ("강의실 90%  →  전체 환경 85%", "#6840A8", True),
            ("수치는 조금 낮아도 적용 환경 범위는 더 넓음", "#4C5560", False),
            ("환경 변화에 대응하는 일반화 성능 향상", "#6840A8", True),
        ],
        COLORS["full200"],
        "#F8F5FF",
    )

    merge_center = (0.50, 0.49)
    merge = plt.Circle(
        merge_center,
        0.062,
        facecolor="white",
        edgecolor="#5B6570",
        linewidth=2,
        zorder=4,
    )
    merge.set_path_effects(
        [pe.SimplePatchShadow(offset=(2, -2), alpha=0.18), pe.Normal()]
    )
    flow.add_patch(merge)
    flow.text(
        *merge_center,
        "+",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#4B5560",
        zorder=5,
    )
    for start, end, rad, color in [
        ((0.32, 0.74), (0.44, 0.515), 0.10, COLORS["location1"]),
        ((0.32, 0.27), (0.44, 0.465), -0.10, COLORS["location2"]),
    ]:
        flow.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=17,
                linewidth=2.3,
                color=color,
            )
        )
    flow.add_patch(
        FancyArrowPatch(
            (0.562, 0.49),
            (0.665, 0.49),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.6,
            color=COLORS["full200"],
        )
    )
    flow.text(
        0.50,
        0.63,
        "두 환경 결합",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#505A64",
    )

    # Add percentage labels after layout positions are finalized.
    figure.canvas.draw()
    for raw_axis, model_axis, gain, color in bottom_pairs:
        raw_pos = raw_axis.get_position()
        model_pos = model_axis.get_position()
        figure.text(
            (raw_pos.x1 + model_pos.x0) / 2,
            max(raw_pos.y1, model_pos.y1) + 0.082,
            f"+{gain:g}%p",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color},
        )

    bottom_y0 = min(pair[0].get_position().y0 for pair in bottom_pairs) - 0.008
    bottom_y1 = max(pair[0].get_position().y1 for pair in bottom_pairs) + 0.10
    for left_pair, right_pair in zip(bottom_pairs[:-1], bottom_pairs[1:]):
        left_edge = left_pair[1].get_position().x1
        right_edge = right_pair[0].get_position().x0
        separator_x = (left_edge + right_edge) / 2
        figure.add_artist(
            Line2D(
                [separator_x, separator_x],
                [bottom_y0, bottom_y1],
                transform=figure.transFigure,
                color="#B8C0C9",
                linewidth=1.5,
                linestyle=(0, (4, 4)),
                alpha=0.9,
            )
        )

    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_csv(items: list[dict], output: Path) -> None:
    rows = []
    for item in items:
        metric = item["report"]["fine_tuned_binary_tsstg"]
        rows.append(
            {
                "model": item["name"],
                "evaluation_scope": item["scope"],
                "samples": item["report"]["samples"],
                **{key: metric[key] for key, _ in METRICS},
            }
        )
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set_style()
    items = load_reports()
    plot_panels(items, OUTPUT / "02_own_scope_performance.png")
    plot_annotated_story(items, OUTPUT / "02_own_scope_performance_annotated.png")
    plot_summary_heatmap(items, OUTPUT / "03_own_scope_summary_heatmap.png")
    write_csv(items, OUTPUT / "metrics.csv")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
