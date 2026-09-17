from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "raspberrypi_haniuim"))

from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph  # noqa: E402


BINARY_CLASS_NAMES = ["NON_FALL", "FALL"]
ORIGINAL_CLASS_NAMES = [
    "Standing",
    "Walking",
    "Sitting",
    "Lying Down",
    "Stand up",
    "Sit down",
    "Fall Down",
]
FALL_DOWN_INDEX = ORIGINAL_CLASS_NAMES.index("Fall Down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original 7-class TSSTG and the fine-tuned binary "
            "TSSTG on exactly the same binary dataset splits."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--original-weights",
        type=Path,
        default=(
            REPO_ROOT
            / "raspberrypi_haniuim"
            / "Models"
            / "TSSTG"
            / "tsstg-model.pth"
        ),
    )
    parser.add_argument("--binary-weights", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["test"],
        help="One or more NPZ splits to combine for evaluation (default: test)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("run_pilot100/model_comparison")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported weights object in {path}: {type(loaded)}")
    return loaded


def make_model(num_classes: int, weights: Path, device: torch.device):
    model = TwoStreamSpatialTemporalGraph(
        {"strategy": "spatial"}, num_class=num_classes
    ).to(device)
    model.load_state_dict(load_state_dict(weights))
    model.eval()
    return model


def forward_model(model, points: torch.Tensor) -> torch.Tensor:
    motion = points[:, :2, 1:, :] - points[:, :2, :-1, :]
    return model((points, motion))


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        confusion[int(label), int(prediction)] += 1

    tn, fp = confusion[0]
    fn, tp = confusion[1]

    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    return {
        "accuracy": divide(tp + tn, int(confusion.sum())),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "fall_precision": precision,
        "fall_recall": recall,
        "fall_f1": divide(2 * precision * recall, precision + recall),
        "non_fall_specificity": specificity,
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    split_paths = [args.dataset / f"{split}.npz" for split in args.splits]
    feature_parts = []
    label_parts = []
    event_id_parts = []
    split_name_parts = []
    for split, split_path in zip(args.splits, split_paths):
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        with np.load(split_path, allow_pickle=False) as data:
            split_features = data["features"].astype(np.float32, copy=True)
            split_labels = data["labels"].astype(np.int64, copy=True)
            split_event_ids = data["event_ids"].astype(str, copy=True)
        feature_parts.append(split_features)
        label_parts.append(split_labels)
        event_id_parts.append(split_event_ids)
        split_name_parts.append(np.full(len(split_labels), split, dtype=object))

    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    event_ids = np.concatenate(event_id_parts)
    split_names = np.concatenate(split_name_parts)

    windows_path = args.dataset / "windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(windows_path)
    with windows_path.open(newline="", encoding="utf-8-sig") as stream:
        location_by_event = {
            row["event_id"]: row["location"] for row in csv.DictReader(stream)
        }
    missing_locations = sorted(set(event_ids) - set(location_by_event))
    if missing_locations:
        raise ValueError(f"Missing location metadata for {missing_locations}")
    locations = np.asarray([location_by_event[event_id] for event_id in event_ids])

    if features.ndim != 4 or features.shape[1:] != (30, 14, 3):
        raise ValueError(
            f"Expected test features shaped (N, 30, 14, 3), got {features.shape}"
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Test labels must use 0=NON_FALL and 1=FALL")

    original_model = make_model(7, args.original_weights, device)
    binary_model = make_model(2, args.binary_weights, device)

    original_scores_parts = []
    binary_scores_parts = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            points = torch.from_numpy(features[start : start + args.batch_size])
            points = points.permute(0, 3, 1, 2).to(device)
            original_scores_parts.append(
                forward_model(original_model, points).cpu().numpy()
            )
            binary_scores_parts.append(
                forward_model(binary_model, points).cpu().numpy()
            )

    original_scores = np.concatenate(original_scores_parts)
    binary_scores = np.concatenate(binary_scores_parts)
    original_7class_predictions = original_scores.argmax(axis=1)
    original_binary_predictions = (
        original_7class_predictions == FALL_DOWN_INDEX
    ).astype(np.int64)
    binary_predictions = binary_scores.argmax(axis=1).astype(np.int64)

    original_metrics = binary_metrics(labels, original_binary_predictions)
    binary_metrics_result = binary_metrics(labels, binary_predictions)

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, event_id in enumerate(event_ids):
        row = {
            "split": str(split_names[index]),
            "event_id": event_id,
            "location": str(locations[index]),
            "true_label": BINARY_CLASS_NAMES[int(labels[index])],
            "original_7class_prediction": ORIGINAL_CLASS_NAMES[
                int(original_7class_predictions[index])
            ],
            "original_binary_prediction": BINARY_CLASS_NAMES[
                int(original_binary_predictions[index])
            ],
            "original_fall_down_score": float(
                original_scores[index, FALL_DOWN_INDEX]
            ),
            "binary_prediction": BINARY_CLASS_NAMES[int(binary_predictions[index])],
            "binary_non_fall_score": float(binary_scores[index, 0]),
            "binary_fall_score": float(binary_scores[index, 1]),
        }
        for class_index, class_name in enumerate(ORIGINAL_CLASS_NAMES):
            key = "original_score_" + class_name.lower().replace(" ", "_")
            row[key] = float(original_scores[index, class_index])
        rows.append(row)

    split_slug = "_".join(args.splits)
    predictions_path = args.output / f"{split_slug}_model_comparison.csv"
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "device": str(device),
        "evaluated_splits": args.splits,
        "data_files": [str(path.resolve()) for path in split_paths],
        "samples": int(len(labels)),
        "class_order": BINARY_CLASS_NAMES,
        "confusion_matrix_convention": "rows=true, columns=predicted",
        "original_mapping": {
            "FALL": "7-class argmax == Fall Down",
            "NON_FALL": "7-class argmax is any other class",
        },
        "original_7class_tsstg": original_metrics,
        "fine_tuned_binary_tsstg": binary_metrics_result,
        "metrics_by_location": {
            location: {
                "samples": int(mask.sum()),
                "original_7class_tsstg": binary_metrics(
                    labels[mask], original_binary_predictions[mask]
                ),
                "fine_tuned_binary_tsstg": binary_metrics(
                    labels[mask], binary_predictions[mask]
                ),
            }
            for location in sorted(set(locations))
            for mask in [locations == location]
        },
        "accuracy_difference_binary_minus_original": (
            binary_metrics_result["accuracy"] - original_metrics["accuracy"]
        ),
    }
    report_path = args.output / "comparison_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"predictions={predictions_path.resolve()}")
    print(f"metrics={report_path.resolve()}")


if __name__ == "__main__":
    main()
