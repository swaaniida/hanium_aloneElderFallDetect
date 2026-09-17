from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent if SOURCE_DIR.name == "tsstg_pipeline" else SOURCE_DIR
sys.path.insert(0, str(REPO_ROOT / "raspberrypi_haniuim"))

from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph  # noqa: E402


CLASS_NAMES = ["NON_FALL", "FALL"]
LEFT_RIGHT_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


class PoseDataset(Dataset):
    def __init__(
        self,
        path: Path,
        augment: bool = False,
        temporal_dropout_probability: float = 0.0,
        temporal_dropout_max_frames: int = 0,
    ) -> None:
        with np.load(path, allow_pickle=False) as data:
            # Explicit copies keep the arrays valid after the NPZ archive closes
            # (required by newer NumPy versions when dtype conversion is a no-op).
            self.features = data["features"].astype(np.float32, copy=True)
            self.labels = data["labels"].astype(np.int64, copy=True)
            self.event_ids = data["event_ids"].astype(str, copy=True)
        self.augment = augment
        self.temporal_dropout_probability = temporal_dropout_probability
        self.temporal_dropout_max_frames = temporal_dropout_max_frames

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        points = self.features[index].copy()
        if self.augment and random.random() < 0.5:
            points[:, :, 0] *= -1.0
            for left, right in LEFT_RIGHT_PAIRS:
                points[:, [left, right]] = points[:, [right, left]]
        if (
            self.augment
            and self.temporal_dropout_max_frames > 0
            and random.random() < self.temporal_dropout_probability
        ):
            drop_frames = random.randint(1, self.temporal_dropout_max_frames)
            start = random.randint(0, len(points) - drop_frames)
            stop = start + drop_frames

            # Simulate a short detector outage. Coordinates are filled exactly as the
            # dataset builder fills missing poses, while confidence marks the gap.
            if start > 0 and stop < len(points):
                for offset, frame_index in enumerate(range(start, stop), start=1):
                    ratio = offset / (drop_frames + 1)
                    points[frame_index, :, :2] = (
                        (1.0 - ratio) * points[start - 1, :, :2]
                        + ratio * points[stop, :, :2]
                    )
            elif start > 0:
                points[start:stop, :, :2] = points[start - 1, :, :2]
            elif stop < len(points):
                points[start:stop, :, :2] = points[stop, :, :2]
            points[start:stop, :, 2] = 0.0
        if self.augment:
            points[:, :, :2] += np.random.normal(
                0.0, 0.005, points[:, :, :2].shape
            ).astype(np.float32)
        return (
            torch.from_numpy(points).permute(2, 0, 1),
            torch.tensor(self.labels[index], dtype=torch.long),
            self.event_ids[index],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def forward_model(model, points):
    motion = points[:, :2, 1:, :] - points[:, :2, :-1, :]
    return model((points, motion))


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for points, labels, _ in loader:
        points = points.to(device)
        labels = labels.to(device)
        targets = torch.nn.functional.one_hot(labels, 2).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = forward_model(model, points)
            loss = loss_function(output, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss) * len(labels)
        correct += int((output.argmax(1) == labels).sum())
        total += len(labels)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    rows = []
    confusion = np.zeros((2, 2), dtype=np.int64)
    with torch.inference_mode():
        for points, labels, event_ids in loader:
            output = forward_model(model, points.to(device)).cpu().numpy()
            predictions = output.argmax(1)
            for event_id, label, prediction, probability in zip(
                event_ids, labels.numpy(), predictions, output
            ):
                confusion[label, prediction] += 1
                rows.append({
                    "event_id": event_id,
                    "true_label": CLASS_NAMES[label],
                    "predicted_label": CLASS_NAMES[prediction],
                    "non_fall_score": float(probability[0]),
                    "fall_score": float(probability[1]),
                })
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    return accuracy, confusion, rows


def load_pretrained_backbone(model, path: Path) -> int:
    if not path.exists():
        print(f"Pretrained weights not found; training from scratch: {path}")
        return 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(current)} compatible tensors")
    return len(compatible)


def train_phase(
    model,
    train_loader,
    validation_loader,
    device,
    epochs,
    learning_rate,
    patience,
):
    if epochs <= 0:
        return deepcopy(model.state_dict()), []
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loss_function = torch.nn.BCELoss()
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_function, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.3f} val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.3f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
    return best_state, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tsstg_pilot_run"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=(
            REPO_ROOT
            / "raspberrypi_haniuim"
            / "Models"
            / "TSSTG"
            / "tsstg-model.pth"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temporal-dropout-probability",
        type=float,
        default=0.35,
        help="Probability of masking one contiguous pose-frame span per train sample",
    )
    parser.add_argument(
        "--temporal-dropout-max-frames",
        type=int,
        default=4,
        help="Maximum contiguous frames masked by temporal pose dropout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.temporal_dropout_probability <= 1.0:
        raise ValueError("--temporal-dropout-probability must be between 0 and 1")
    if not 0 <= args.temporal_dropout_max_frames < 30:
        raise ValueError("--temporal-dropout-max-frames must be between 0 and 29")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_set = PoseDataset(
        args.dataset / "train.npz",
        augment=True,
        temporal_dropout_probability=args.temporal_dropout_probability,
        temporal_dropout_max_frames=args.temporal_dropout_max_frames,
    )
    validation_set = PoseDataset(args.dataset / "validation.npz")
    test_set = PoseDataset(args.dataset / "test.npz")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, num_workers=0
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

    model = TwoStreamSpatialTemporalGraph(
        {"strategy": "spatial"}, num_class=2
    ).to(device)
    loaded_tensors = load_pretrained_backbone(model, args.pretrained)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fcn.parameters():
        parameter.requires_grad = True
    head_state, head_history = train_phase(
        model,
        train_loader,
        validation_loader,
        device,
        args.head_epochs,
        1e-3,
        args.patience,
    )
    model.load_state_dict(head_state)

    for parameter in model.parameters():
        parameter.requires_grad = True
    final_state, finetune_history = train_phase(
        model,
        train_loader,
        validation_loader,
        device,
        args.finetune_epochs,
        1e-4,
        args.patience,
    )
    model.load_state_dict(final_state)

    validation_accuracy, validation_confusion, _ = evaluate(
        model, validation_loader, device
    )
    test_accuracy, test_confusion, test_rows = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), args.output / "binary_tsstg_state_dict.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "window_frames": 30,
            "pose_hz": 25.0,
            "augmentation": {
                "horizontal_flip_probability": 0.5,
                "xy_gaussian_noise_std": 0.005,
                "temporal_dropout_probability": args.temporal_dropout_probability,
                "temporal_dropout_max_frames": args.temporal_dropout_max_frames,
            },
        },
        args.output / "binary_tsstg_checkpoint.pth",
    )

    with (args.output / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    metrics = {
        "device": str(device),
        "pretrained_tensors_loaded": loaded_tensors,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "validation_confusion_matrix": validation_confusion.tolist(),
        "test_confusion_matrix": test_confusion.tolist(),
        "class_names": CLASS_NAMES,
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "xy_gaussian_noise_std": 0.005,
            "temporal_dropout_probability": args.temporal_dropout_probability,
            "temporal_dropout_max_frames": args.temporal_dropout_max_frames,
        },
        "head_history": head_history,
        "finetune_history": finetune_history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metrics.items() if "history" not in k}, indent=2))


if __name__ == "__main__":
    main()
