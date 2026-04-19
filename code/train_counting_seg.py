from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


CLASS_NAMES = ["Type1", "Type2", "Type3", "Type4", "Type5"]


def parse_args() -> argparse.Namespace:
    repo_hw4 = Path(__file__).resolve().parents[1]
    default_model = repo_hw4.parent / "yolov8s-seg.pt"
    if not default_model.exists():
        default_model = Path("yolov8s-seg.pt")
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8s-seg model for five-class screw instance segmentation."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=repo_hw4 / "dataset" / "counting",
        help="Dataset root containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=repo_hw4 / "model" / "counting",
        help="Directory where training runs and exported weights are saved.",
    )
    parser.add_argument("--model", type=str, default=str(default_model))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--run_name",
        type=str,
        default="train",
        help="Subdirectory name under output_dir/runs.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def write_dataset_yaml(dataset_dir: Path) -> Path:
    train_images = dataset_dir / "train" / "images" / "train"
    val_images = dataset_dir / "val" / "images" / "val"
    train_labels = dataset_dir / "train" / "labels" / "train"
    val_labels = dataset_dir / "val" / "labels" / "val"

    ensure_exists(train_images, "training images directory")
    ensure_exists(val_images, "validation images directory")
    ensure_exists(train_labels, "training labels directory")
    ensure_exists(val_labels, "validation labels directory")

    yaml_path = dataset_dir / "data.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.as_posix()}",
                "train: train/images/train",
                "val: val/images/val",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def copy_best_weights(run_dir: Path, output_dir: Path) -> None:
    weights_dir = run_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if best.exists():
        shutil.copy2(best, output_dir / "best.pt")
    if last.exists():
        shutil.copy2(last, output_dir / "last.pt")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / ".ultralytics"))

    data_yaml = write_dataset_yaml(dataset_dir)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Install it with: pip install ultralytics"
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(output_dir / "runs"),
        "name": args.run_name,
        "task": "segment",
        "exist_ok": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    run_dir = output_dir / "runs" / args.run_name
    copy_best_weights(run_dir, output_dir)

    print(f"dataset yaml: {data_yaml}")
    print(f"training run: {run_dir}")
    print(f"best model: {output_dir / 'best.pt'}")
    print(results)


if __name__ == "__main__":
    main()
