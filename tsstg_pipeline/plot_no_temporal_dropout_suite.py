from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .plot_seed42_model_suite import (
    COLORS,
    METRICS,
    load_json,
    plot_common_test,
    plot_location_heatmaps,
    plot_training_suite,
    set_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/seed42_no_temporal_dropout"),
    )
    return parser.parse_args()


def write_csv(models: list[tuple[str, dict, str]], output: Path) -> None:
    rows = [
        {
            "model": name,
            **{key: metric[key] for key, _ in METRICS},
        }
        for name, metric, _ in models
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    paths = {
        "location1": Path("run_location1_nodropout_seed42"),
        "location2": Path("run_location2_nodropout_seed42"),
        "full200": Path("run_full200_nodropout_seed42"),
    }
    training = {key: load_json(path / "metrics.json") for key, path in paths.items()}
    plot_training_suite(
        training,
        args.output / "01_no_dropout_training_curves.png",
        title="Seed 42 신규 TSSTG 학습 과정 — Temporal Pose Dropout 미적용",
    )

    for scope, subdir in [
        ("test20", "common_test_comparison"),
        ("test40", "test40_comparison"),
    ]:
        output = args.output / scope
        output.mkdir(parents=True, exist_ok=True)
        reports = {
            key: load_json(path / subdir / "comparison_metrics.json")
            for key, path in paths.items()
        }
        raw = reports["full200"]["original_7class_tsstg"]
        models = [
            ("Raw TSSTG", raw, COLORS["raw"]),
            (
                "위치 1 (Dropout 미적용)",
                reports["location1"]["fine_tuned_binary_tsstg"],
                COLORS["location1"],
            ),
            (
                "위치 2 (Dropout 미적용)",
                reports["location2"]["fine_tuned_binary_tsstg"],
                COLORS["location2"],
            ),
            (
                "전체 결합 (Dropout 미적용)",
                reports["full200"]["fine_tuned_binary_tsstg"],
                COLORS["full200"],
            ),
        ]
        test_samples = int(reports["full200"]["samples"])
        per_location = int(
            reports["full200"]["metrics_by_location"]["location_1"]["samples"]
        )
        plot_common_test(
            models,
            output / "02_no_dropout_model_performance.png",
            test_samples,
        )
        plot_location_heatmaps(
            [
                ("위치 1 (미적용)", reports["location1"]),
                ("위치 2 (미적용)", reports["location2"]),
                ("전체 결합 (미적용)", reports["full200"]),
            ],
            output / "03_no_dropout_location_heatmap.png",
            per_location,
        )
        write_csv(models, output / "metrics.csv")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
