from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_NAMES = ["Type1", "Type2", "Type3", "Type4", "Type5"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
VIDEO_SAMPLING_BACKENDS = ("sequential", "grab", "seek")
BENCHMARK_VIDEO_NAMES = ("IMG_2374.MOV", "IMG_2375.MOV", "IMG_2376.MOV")


def repo_hw4_root() -> Path:
    return Path(__file__).resolve().parents[1]


def preferred_model_path() -> Path:
    repo_hw4 = repo_hw4_root()
    candidates = [
        repo_hw4 / "model" / "counting_v1v2_80_v3first10_prev_train" / "best.pt",
        repo_hw4 / "model" / "counting" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class GlobalInstance:
    instance_id: int
    global_center: np.ndarray
    center_weight: float = 0.0
    class_votes: np.ndarray = field(default_factory=lambda: np.zeros(len(CLASS_NAMES), dtype=np.float32))
    best_confidence: float = 0.0
    best_class_id: int = 0
    observations: int = 0
    quality_observations: int = 0
    seen_frames: list[int] = field(default_factory=list)
    area_min: float = 0.0
    area_max: float = 0.0
    area_sum: float = 0.0
    observation_log: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        detection: dict[str, Any],
        frame_index: int,
        global_center: np.ndarray,
        is_quality: bool,
        registration_failed: bool,
        save_observation_log: bool = False,
    ) -> None:
        confidence = float(detection["confidence"])
        class_id = int(detection["class_id"])
        area = float(detection["mask_area"])

        if self.center_weight <= 0:
            self.global_center = global_center.copy()
            self.center_weight = max(confidence, 1e-3)
        else:
            weight = max(confidence, 1e-3)
            self.global_center = (self.global_center * self.center_weight + global_center * weight) / (
                self.center_weight + weight
            )
            self.center_weight += weight

        self.observations += 1
        if is_quality:
            self.quality_observations += 1
            self.class_votes[class_id] += confidence
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_class_id = class_id

        self.seen_frames.append(frame_index)
        self.area_sum += area
        self.area_min = area if self.area_min == 0.0 else min(self.area_min, area)
        self.area_max = max(self.area_max, area)
        if save_observation_log:
            self.observation_log.append(
                {
                    "frame": frame_index,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": confidence,
                    "bbox": detection["bbox"],
                    "centroid": detection["centroid"],
                    "global_center": [float(global_center[0]), float(global_center[1])],
                    "mask_area": area,
                    "quality": bool(is_quality),
                    "registration_failed": bool(registration_failed),
                }
            )

    def dominant_class_id(self) -> int:
        if float(self.class_votes.sum()) > 0:
            return int(np.argmax(self.class_votes))
        return int(self.best_class_id)

    def mean_area(self) -> float:
        return self.area_sum / max(1, self.observations)

    def to_json(self) -> dict[str, Any]:
        class_id = self.dominant_class_id()
        item = {
            "instance_id": int(self.instance_id),
            "global_center": [float(self.global_center[0]), float(self.global_center[1])],
            "final_class_id": class_id,
            "final_class_name": CLASS_NAMES[class_id],
            "class_votes": [float(value) for value in self.class_votes.tolist()],
            "best_confidence": float(self.best_confidence),
            "observations": int(self.observations),
            "quality_observations": int(self.quality_observations),
            "seen_frames": self.seen_frames,
            "mask_area_stats": {
                "min": float(self.area_min),
                "max": float(self.area_max),
                "mean": float(self.mean_area()),
            },
        }
        if self.observation_log:
            item["observation_log"] = self.observation_log
        return item


@dataclass
class RegistrationFeatures:
    gray: np.ndarray
    keypoints: list[Any] = field(default_factory=list)
    descriptors: np.ndarray | None = None
    orb_ready: bool = False


def parse_args() -> argparse.Namespace:
    repo_hw4 = repo_hw4_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained YOLOv8 segmentation model. Image inputs are counted "
            "per frame; video inputs are counted with sampled-frame global deduplication."
        )
    )
    parser.add_argument("--model_path", type=Path, default=preferred_model_path())
    parser.add_argument("--source", type=Path, help="Image file, image folder, or video file.")
    parser.add_argument("--output_dir", type=Path, default=repo_hw4 / "output" / "counting")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--predict_batch", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--save_vis", action="store_true", help="Save annotated visualizations.")
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--video_sampling_backend", choices=VIDEO_SAMPLING_BACKENDS, default="grab")
    parser.add_argument(
        "--benchmark_video_sampling",
        action="store_true",
        help="Benchmark sequential/grab/seek video sampling on the three bundled sample videos.",
    )
    parser.add_argument("--registration_scale", type=float, default=0.2)
    parser.add_argument("--registration_backend", choices=["auto", "lk", "orb"], default="orb")
    parser.add_argument("--min_h_inliers", type=int, default=25)
    parser.add_argument("--global_match_radius_ratio", type=float, default=0.03)
    parser.add_argument("--global_cluster_radius_ratio", type=float, default=0.015)
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--quality_conf", type=float, default=0.35)
    parser.add_argument("--border_margin", type=float, default=0.03)
    parser.add_argument("--roi_margin", type=float, default=0.05)
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.00005)
    parser.add_argument("--save_observation_log", action="store_true")
    return parser.parse_args()


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.name,
    )


def list_video_files(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES],
        key=lambda path: path.name,
    )


def build_inference_args(
    *,
    model_path: Path | None = None,
    source: Path | None = None,
    output_dir: Path | None = None,
    imgsz: int = 960,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
    predict_batch: int | None = None,
    max_frames: int = 300,
    save_vis: bool = False,
    frame_stride: int = 10,
    video_sampling_backend: str = "grab",
    benchmark_video_sampling: bool = False,
    registration_scale: float = 0.2,
    registration_backend: str = "orb",
    min_h_inliers: int = 25,
    global_match_radius_ratio: float = 0.03,
    global_cluster_radius_ratio: float = 0.015,
    min_observations: int = 2,
    quality_conf: float = 0.35,
    border_margin: float = 0.03,
    roi_margin: float = 0.05,
    min_mask_area_ratio: float = 0.00005,
    save_observation_log: bool = False,
) -> argparse.Namespace:
    repo_hw4 = repo_hw4_root()
    return argparse.Namespace(
        model_path=Path(model_path) if model_path is not None else preferred_model_path(),
        source=Path(source) if source is not None else None,
        output_dir=Path(output_dir) if output_dir is not None else repo_hw4 / "output" / "counting",
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        device=device,
        predict_batch=predict_batch,
        max_frames=int(max_frames),
        save_vis=bool(save_vis),
        frame_stride=int(frame_stride),
        video_sampling_backend=video_sampling_backend,
        benchmark_video_sampling=bool(benchmark_video_sampling),
        registration_scale=float(registration_scale),
        registration_backend=registration_backend,
        min_h_inliers=int(min_h_inliers),
        global_match_radius_ratio=float(global_match_radius_ratio),
        global_cluster_radius_ratio=float(global_cluster_radius_ratio),
        min_observations=int(min_observations),
        quality_conf=float(quality_conf),
        border_margin=float(border_margin),
        roi_margin=float(roi_margin),
        min_mask_area_ratio=float(min_mask_area_ratio),
        save_observation_log=bool(save_observation_log),
    )


def load_model(model_path: Path) -> Any:
    repo_hw4 = repo_hw4_root()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo_hw4 / "model" / "counting" / ".ultralytics"))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install it with: pip install ultralytics") from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if model_path.suffix.lower() == ".engine":
        return YOLO(str(model_path), task="segment")
    return YOLO(str(model_path))


def mask_centroid(mask: np.ndarray | None, bbox: list[float]) -> list[float]:
    if mask is not None and mask.size > 0:
        ys, xs = np.where(mask > 0.5)
        if len(xs) > 0:
            return [float(xs.mean()), float(ys.mean())]
    x1, y1, x2, y2 = bbox
    return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]


def mask_area(mask: np.ndarray | None, bbox: list[float]) -> float:
    if mask is not None and mask.size > 0:
        return float(np.count_nonzero(mask > 0.5))
    x1, y1, x2, y2 = bbox
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def polygon_centroid_and_area(polygon: np.ndarray | None, bbox: list[float]) -> tuple[list[float], float]:
    if polygon is None or len(polygon) < 3:
        x1, y1, x2, y2 = bbox
        return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)], float(
            max(0.0, x2 - x1) * max(0.0, y2 - y1)
        )

    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x_values = points[:, 0]
    y_values = points[:, 1]
    shifted_x = np.roll(x_values, -1)
    shifted_y = np.roll(y_values, -1)
    cross = x_values * shifted_y - shifted_x * y_values
    signed_area = float(cross.sum() * 0.5)
    area = abs(signed_area)
    if area < 1e-6:
        return [float(x_values.mean()), float(y_values.mean())], 0.0

    centroid_x = float(((x_values + shifted_x) * cross).sum() / (6.0 * signed_area))
    centroid_y = float(((y_values + shifted_y) * cross).sum() / (6.0 * signed_area))
    return [centroid_x, centroid_y], area


def near_border(bbox: list[float], width: int, height: int, margin: float) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        x1 <= margin * width
        or y1 <= margin * height
        or x2 >= (1.0 - margin) * width
        or y2 >= (1.0 - margin) * height
    )


def in_center_roi(center: list[float], width: int, height: int, margin: float) -> bool:
    x, y = center
    return margin * width <= x <= (1.0 - margin) * width and margin * height <= y <= (1.0 - margin) * height


def is_quality_detection(detection: dict[str, Any], width: int, height: int, args: argparse.Namespace) -> bool:
    frame_area = float(width * height)
    return (
        float(detection["confidence"]) >= args.quality_conf
        and float(detection["mask_area"]) >= args.min_mask_area_ratio * frame_area
        and not near_border(detection["bbox"], width, height, args.border_margin)
        and in_center_roi(detection["centroid"], width, height, args.roi_margin)
    )


def draw_instances(
    image: np.ndarray,
    instances: list[dict[str, Any]],
    instance_ids: list[int] | None = None,
) -> np.ndarray:
    overlay = image.copy()
    output = image.copy()
    colors = [(255, 80, 80), (80, 220, 80), (80, 150, 255), (220, 180, 60), (200, 80, 220)]

    for instance in instances:
        class_id = int(instance["class_id"])
        color = colors[class_id % len(colors)]
        polygon = instance.get("_polygon")
        if polygon is not None and len(polygon) >= 3:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [points], color)

    output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

    for idx, instance in enumerate(instances):
        class_id = int(instance["class_id"])
        color = colors[class_id % len(colors)]
        x1, y1, x2, y2 = [int(round(value)) for value in instance["bbox"]]
        polygon = instance.get("_polygon")
        if polygon is not None and len(polygon) >= 3:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [points], True, color, 2)
        label = f'{instance["class_name"]} {instance["confidence"]:.2f}'
        if instance_ids is not None and idx < len(instance_ids) and instance_ids[idx] >= 0:
            label = f'G{instance_ids[idx]} {label}'
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return output


def result_to_instances(result: Any) -> tuple[list[dict[str, Any]], list[int]]:
    instances: list[dict[str, Any]] = []
    counts = [0] * len(CLASS_NAMES)
    if result.boxes is None or len(result.boxes) == 0:
        return instances, counts

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    polygons = result.masks.xy if result.masks is not None and result.masks.xy is not None else None

    for idx, (bbox, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue
        polygon = polygons[idx] if polygons is not None and idx < len(polygons) else None
        bbox_list = [float(value) for value in bbox.tolist()]
        centroid, area = polygon_centroid_and_area(polygon, bbox_list)
        instances.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(confidence),
                "bbox": bbox_list,
                "centroid": centroid,
                "mask_area": area,
                "_polygon": polygon,
            }
        )
        counts[class_id] += 1
    return instances, counts


def strip_internal_masks(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for instance in instances:
        item = dict(instance)
        item.pop("_mask", None)
        item.pop("_polygon", None)
        clean.append(item)
    return clean


def summarize_image_items(per_item: dict[str, dict[str, Any]], source: str) -> dict[str, Any]:
    totals = [0] * len(CLASS_NAMES)
    for item in per_item.values():
        totals = [left + int(right) for left, right in zip(totals, item["counts"])]
    return {"mode": "image", "source": source, "class_names": CLASS_NAMES, "total_counts": totals, "items": per_item}


def save_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "counts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if summary["mode"] == "video":
            writer.writerow(["video", *CLASS_NAMES])
            writer.writerow([summary["source"], *summary["total_counts"]])
        else:
            writer.writerow(["item", *CLASS_NAMES])
            for item_name, item in summary["items"].items():
                writer.writerow([item_name, *item["counts"]])
            writer.writerow(["TOTAL", *summary["total_counts"]])


def predict_image(model: Any, image_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray | None]:
    result = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]
    instances, counts = result_to_instances(result)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if args.save_vis else None
    vis = draw_instances(image, instances) if image is not None else None
    return {"counts": counts, "instances": strip_internal_masks(instances)}, vis


def build_sample_frame_indices(actual_last_frame: int, frame_stride: int) -> list[int]:
    if actual_last_frame <= 0:
        return []
    indices = list(range(1, actual_last_frame + 1, frame_stride))
    if not indices or indices[-1] != actual_last_frame:
        indices.append(actual_last_frame)
    return indices


def sample_video_frames_sequential(
    video_path: Path,
    max_frames: int,
    frame_stride: int,
) -> tuple[list[tuple[int, np.ndarray]], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    samples: list[tuple[int, np.ndarray]] = []
    frame_index = 1
    last_frame: np.ndarray | None = None
    last_index = 0
    try:
        while frame_index <= max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            last_frame = frame
            last_index = frame_index
            if (frame_index - 1) % frame_stride == 0:
                samples.append((frame_index, frame.copy()))
            frame_index += 1
    finally:
        cap.release()

    if last_frame is not None and last_index > 0:
        if not samples or samples[-1][0] != last_index:
            samples.append((last_index, last_frame.copy()))
    return samples, last_index


def sample_video_frames_grab(
    video_path: Path,
    max_frames: int,
    frame_stride: int,
) -> tuple[list[tuple[int, np.ndarray]], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    samples: list[tuple[int, np.ndarray]] = []
    frame_index = 1
    last_index = 0
    try:
        while frame_index <= max_frames:
            ok = cap.grab()
            if not ok:
                break
            last_index = frame_index
            if (frame_index - 1) % frame_stride == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                samples.append((frame_index, frame.copy()))
            frame_index += 1

        if last_index > 0 and (not samples or samples[-1][0] != last_index):
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_index - 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                samples.append((last_index, frame.copy()))
    finally:
        cap.release()

    return samples, last_index


def sample_video_frames_seek(
    video_path: Path,
    max_frames: int,
    frame_stride: int,
) -> tuple[list[tuple[int, np.ndarray]], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            return sample_video_frames_sequential(video_path, max_frames, frame_stride)

        actual_last_frame = min(max_frames, frame_count)
        samples: list[tuple[int, np.ndarray]] = []
        for frame_index in build_sample_frame_indices(actual_last_frame, frame_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            samples.append((frame_index, frame.copy()))
        return samples, actual_last_frame
    finally:
        cap.release()


def sample_video_frames(
    video_path: Path,
    max_frames: int,
    frame_stride: int,
    backend: str = "sequential",
) -> tuple[list[tuple[int, np.ndarray]], int]:
    if backend == "sequential":
        return sample_video_frames_sequential(video_path, max_frames, frame_stride)
    if backend == "grab":
        return sample_video_frames_grab(video_path, max_frames, frame_stride)
    if backend == "seek":
        return sample_video_frames_seek(video_path, max_frames, frame_stride)
    raise ValueError(f"Unsupported video sampling backend: {backend}")


def samples_equal(
    reference_samples: list[tuple[int, np.ndarray]],
    reference_last: int,
    candidate_samples: list[tuple[int, np.ndarray]],
    candidate_last: int,
) -> dict[str, bool]:
    same_actual_last_frame = int(reference_last) == int(candidate_last)
    reference_indices = [frame_index for frame_index, _frame in reference_samples]
    candidate_indices = [frame_index for frame_index, _frame in candidate_samples]
    same_sampled_frames = reference_indices == candidate_indices
    same_frame_pixels = False
    if same_sampled_frames and len(reference_samples) == len(candidate_samples):
        same_frame_pixels = all(
            np.array_equal(reference_frame, candidate_frame)
            for (_reference_index, reference_frame), (_candidate_index, candidate_frame) in zip(
                reference_samples, candidate_samples
            )
        )
    return {
        "same_actual_last_frame": same_actual_last_frame,
        "same_sampled_frames": same_sampled_frames,
        "same_frame_pixels": same_frame_pixels,
        "equivalent_to_reference": same_actual_last_frame and same_sampled_frames and same_frame_pixels,
    }


def benchmark_video_sampling(args: argparse.Namespace) -> dict[str, Any]:
    repo_hw4 = Path(__file__).resolve().parents[1]
    video_paths = [repo_hw4 / "vedio_exp" / "vedio_exp" / video_name for video_name in BENCHMARK_VIDEO_NAMES]
    missing = [str(path) for path in video_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Benchmark videos not found: {missing}")

    summary: dict[str, Any] = {
        "mode": "video_sampling_benchmark",
        "settings": {
            "max_frames": int(args.max_frames),
            "frame_stride": int(args.frame_stride),
            "backends": list(VIDEO_SAMPLING_BACKENDS),
        },
        "videos": {},
    }

    aggregate_times = {backend: [] for backend in VIDEO_SAMPLING_BACKENDS}
    valid_global_candidates = set(VIDEO_SAMPLING_BACKENDS)

    for video_path in video_paths:
        reference_samples: list[tuple[int, np.ndarray]] | None = None
        reference_last = 0
        video_results: dict[str, Any] = {}

        for backend in VIDEO_SAMPLING_BACKENDS:
            start_time = time.perf_counter()
            samples, actual_last_frame = sample_video_frames(video_path, args.max_frames, args.frame_stride, backend)
            elapsed = time.perf_counter() - start_time
            backend_result: dict[str, Any] = {
                "elapsed_seconds": elapsed,
                "sample_count": len(samples),
                "actual_last_frame": int(actual_last_frame),
                "sampled_frames": [frame_index for frame_index, _frame in samples],
            }
            if backend == "sequential":
                backend_result.update(
                    {
                        "same_actual_last_frame": True,
                        "same_sampled_frames": True,
                        "same_frame_pixels": True,
                        "equivalent_to_reference": True,
                    }
                )
                reference_samples = samples
                reference_last = actual_last_frame
            else:
                assert reference_samples is not None
                backend_result.update(samples_equal(reference_samples, reference_last, samples, actual_last_frame))
            video_results[backend] = backend_result
            aggregate_times[backend].append(elapsed)

        equivalent_backends = [
            backend for backend, result in video_results.items() if bool(result["equivalent_to_reference"])
        ]
        winner = min(equivalent_backends, key=lambda item: float(video_results[item]["elapsed_seconds"]))
        video_results["winner"] = winner
        summary["videos"][video_path.name] = video_results
        valid_global_candidates &= set(equivalent_backends)

    if valid_global_candidates:
        global_winner = min(
            sorted(valid_global_candidates),
            key=lambda backend: sum(aggregate_times[backend]) / max(len(aggregate_times[backend]), 1),
        )
    else:
        global_winner = "sequential"

    summary["aggregate"] = {
        backend: {
            "average_elapsed_seconds": sum(aggregate_times[backend]) / max(len(aggregate_times[backend]), 1),
            "video_count": len(aggregate_times[backend]),
        }
        for backend in VIDEO_SAMPLING_BACKENDS
    }
    summary["global_winner"] = global_winner

    output_path = args.output_dir / "sampling_benchmark.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_batched_predictions(
    model: Any,
    samples: list[tuple[int, np.ndarray]],
    args: argparse.Namespace,
    batch_size: int = 8,
) -> list[tuple[int, np.ndarray, Any]]:
    results: list[tuple[int, np.ndarray, Any]] = []
    effective_batch_size = int(args.predict_batch or batch_size)
    for start in range(0, len(samples), effective_batch_size):
        batch = samples[start : start + effective_batch_size]
        frames = [frame for _, frame in batch]
        preds = model.predict(
            source=frames,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=len(batch),
            verbose=False,
        )
        for (frame_index, frame), pred in zip(batch, preds):
            results.append((frame_index, frame, pred))
    return results


def build_registration_features(frame: np.ndarray, scale: float) -> RegistrationFeatures:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return RegistrationFeatures(gray=gray)


def ensure_orb_features(features: RegistrationFeatures) -> RegistrationFeatures:
    if not features.orb_ready:
        orb = cv2.ORB_create(nfeatures=1200, fastThreshold=5)
        keypoints, descriptors = orb.detectAndCompute(features.gray, None)
        features.keypoints = keypoints or []
        features.descriptors = descriptors
        features.orb_ready = True
    return features


def scale_homography(h_small: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return h_small
    scale_matrix = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    inv_scale = np.linalg.inv(scale_matrix)
    return inv_scale @ h_small @ scale_matrix


def estimate_homography_lk(
    prev_features: RegistrationFeatures,
    curr_features: RegistrationFeatures,
    scale: float,
    min_h_inliers: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    status = {
        "success": False,
        "matches": 0,
        "inliers": 0,
        "backend": "lk",
        "message": "",
    }
    prev_points = cv2.goodFeaturesToTrack(
        prev_features.gray,
        maxCorners=900,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if prev_points is None or len(prev_points) < 8:
        status["message"] = "not enough LK source points"
        return None, status

    curr_points, lk_status, _err = cv2.calcOpticalFlowPyrLK(
        prev_features.gray,
        curr_features.gray,
        prev_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if curr_points is None or lk_status is None:
        status["message"] = "LK failed"
        return None, status

    valid = lk_status.reshape(-1) == 1
    prev_tracked = prev_points[valid]
    curr_tracked = curr_points[valid]
    status["matches"] = int(len(curr_tracked))
    if len(curr_tracked) < 8:
        status["message"] = "not enough LK tracks"
        return None, status

    h_small, inlier_mask = cv2.findHomography(curr_tracked, prev_tracked, cv2.RANSAC, 3.0)
    if h_small is None or inlier_mask is None:
        status["message"] = "findHomography failed"
        return None, status

    inliers = int(inlier_mask.sum())
    status["inliers"] = inliers
    if inliers < min_h_inliers:
        status["message"] = "inliers below threshold"
        return None, status

    h_full = scale_homography(h_small, scale)
    if not np.isfinite(h_full).all():
        status["message"] = "non-finite homography"
        return None, status

    status["success"] = True
    status["message"] = "ok"
    return h_full.astype(np.float32), status


def estimate_homography_orb(
    prev_features: RegistrationFeatures,
    curr_features: RegistrationFeatures,
    scale: float,
    min_h_inliers: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    status = {
        "success": False,
        "matches": 0,
        "inliers": 0,
        "backend": "orb",
        "message": "",
    }
    ensure_orb_features(prev_features)
    ensure_orb_features(curr_features)
    prev_keypoints = prev_features.keypoints
    prev_desc = prev_features.descriptors
    curr_keypoints = curr_features.keypoints
    curr_desc = curr_features.descriptors
    if prev_desc is None or curr_desc is None:
        status["message"] = "missing descriptors"
        return None, status

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(curr_desc, prev_desc, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)
    status["matches"] = len(good_matches)

    if len(good_matches) < 8:
        status["message"] = "not enough matches"
        return None, status

    curr_points = np.float32([curr_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    prev_points = np.float32([prev_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    h_small, inlier_mask = cv2.findHomography(curr_points, prev_points, cv2.RANSAC, 3.0)
    if h_small is None or inlier_mask is None:
        status["message"] = "findHomography failed"
        return None, status

    inliers = int(inlier_mask.sum())
    status["inliers"] = inliers
    if inliers < min_h_inliers:
        status["message"] = "inliers below threshold"
        return None, status

    h_full = scale_homography(h_small, scale)

    if not np.isfinite(h_full).all():
        status["message"] = "non-finite homography"
        return None, status

    status["success"] = True
    status["message"] = "ok"
    return h_full.astype(np.float32), status


def estimate_homography(
    prev_features: RegistrationFeatures,
    curr_features: RegistrationFeatures,
    scale: float,
    min_h_inliers: int,
    backend: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if backend == "lk":
        return estimate_homography_lk(prev_features, curr_features, scale, min_h_inliers)
    if backend == "orb":
        return estimate_homography_orb(prev_features, curr_features, scale, min_h_inliers)

    h_lk, status_lk = estimate_homography_lk(prev_features, curr_features, scale, min_h_inliers)
    if h_lk is not None:
        return h_lk, status_lk
    h_orb, status_orb = estimate_homography_orb(prev_features, curr_features, scale, min_h_inliers)
    status_orb["lk_message"] = status_lk["message"]
    status_orb["lk_inliers"] = int(status_lk["inliers"])
    status_orb["lk_matches"] = int(status_lk["matches"])
    return h_orb, status_orb


def transform_point(transform: np.ndarray, point: list[float]) -> np.ndarray:
    points = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    transformed = cv2.perspectiveTransform(points, transform)
    return transformed.reshape(2).astype(np.float32)


def class_conflict(instance: GlobalInstance, detection: dict[str, Any], quality_conf: float) -> bool:
    if instance.quality_observations < 2 or float(instance.class_votes.sum()) <= 0:
        return False
    dominant = instance.dominant_class_id()
    det_class = int(detection["class_id"])
    det_conf = float(detection["confidence"])
    if det_class == dominant or det_conf < quality_conf:
        return False
    dominant_vote = float(instance.class_votes[dominant])
    det_vote = float(instance.class_votes[det_class])
    return dominant_vote >= max(0.8, 2.0 * max(det_vote, 1e-6))


def area_compatible(instance: GlobalInstance, detection: dict[str, Any]) -> bool:
    mean_area = instance.mean_area()
    if mean_area <= 0:
        return True
    ratio = float(detection["mask_area"]) / mean_area
    return 0.25 <= ratio <= 4.0


def merge_detections_globally(
    frames_data: list[dict[str, Any]],
    image_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[list[GlobalInstance], list[dict[str, Any]]]:
    height, width = image_shape[:2]
    diagonal = float(np.hypot(width, height))
    base_radius = args.global_match_radius_ratio * diagonal
    global_instances: list[GlobalInstance] = []
    registration_status: list[dict[str, Any]] = []
    next_instance_id = 1
    consecutive_failures = 0

    for frame_data in frames_data:
        registration_failed = bool(frame_data["registration_failed"])
        if registration_failed:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        status_item = dict(frame_data["registration_status"])
        status_item["frame"] = int(frame_data["frame"])
        status_item["registration_failed"] = registration_failed
        status_item["skip_new_instances"] = consecutive_failures >= 2
        registration_status.append(status_item)

        used_instances: set[int] = set()
        assigned_ids: list[int] = []
        frame_radius = base_radius * (0.5 if registration_failed else 1.0)

        for detection in frame_data["instances"]:
            global_center = transform_point(frame_data["global_transform"], detection["centroid"])
            detection["global_center"] = [float(global_center[0]), float(global_center[1])]
            quality = is_quality_detection(detection, width, height, args)

            best_match: GlobalInstance | None = None
            best_distance = None
            for instance in global_instances:
                if instance.instance_id in used_instances:
                    continue
                distance = float(np.linalg.norm(global_center - instance.global_center))
                if distance > frame_radius:
                    continue
                if not area_compatible(instance, detection):
                    continue
                if class_conflict(instance, detection, args.quality_conf):
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match = instance

            if best_match is not None:
                best_match.update(
                    detection,
                    frame_data["frame"],
                    global_center,
                    quality,
                    registration_failed,
                    args.save_observation_log,
                )
                used_instances.add(best_match.instance_id)
                assigned_ids.append(best_match.instance_id)
                continue

            if consecutive_failures >= 2:
                assigned_ids.append(-1)
                continue
            if registration_failed and not quality:
                assigned_ids.append(-1)
                continue

            instance = GlobalInstance(instance_id=next_instance_id, global_center=global_center.copy())
            instance.update(
                detection,
                frame_data["frame"],
                global_center,
                quality,
                registration_failed,
                args.save_observation_log,
            )
            global_instances.append(instance)
            used_instances.add(instance.instance_id)
            assigned_ids.append(instance.instance_id)
            next_instance_id += 1

        frame_data["instance_ids"] = assigned_ids

    valid_instances = [
        instance
        for instance in global_instances
        if instance.observations >= args.min_observations and instance.quality_observations >= 1
    ]
    return valid_instances, registration_status


def counts_from_global_instances(instances: list[GlobalInstance]) -> list[int]:
    counts = [0] * len(CLASS_NAMES)
    for instance in instances:
        counts[instance.dominant_class_id()] += 1
    return counts


def merge_global_instance_group(group: list[GlobalInstance]) -> GlobalInstance:
    merged = GlobalInstance(
        instance_id=min(instance.instance_id for instance in group),
        global_center=np.zeros(2, dtype=np.float32),
    )
    total_center_weight = sum(float(instance.center_weight) for instance in group)
    if total_center_weight > 0:
        weighted_center = sum(instance.global_center * float(instance.center_weight) for instance in group)
        merged.global_center = (weighted_center / total_center_weight).astype(np.float32)
        merged.center_weight = total_center_weight
    else:
        merged.global_center = np.mean([instance.global_center for instance in group], axis=0).astype(np.float32)
        merged.center_weight = float(len(group))

    merged.class_votes = sum((instance.class_votes for instance in group), np.zeros(len(CLASS_NAMES), dtype=np.float32))
    merged.best_confidence = max(float(instance.best_confidence) for instance in group)
    best_instance = max(group, key=lambda instance: float(instance.best_confidence))
    merged.best_class_id = int(best_instance.best_class_id)
    merged.observations = sum(int(instance.observations) for instance in group)
    merged.quality_observations = sum(int(instance.quality_observations) for instance in group)
    merged.seen_frames = sorted({frame for instance in group for frame in instance.seen_frames})
    merged.area_min = min(float(instance.area_min) for instance in group)
    merged.area_max = max(float(instance.area_max) for instance in group)
    merged.area_sum = sum(float(instance.area_sum) for instance in group)
    merged.observation_log = sorted(
        [item for instance in group for item in instance.observation_log],
        key=lambda item: (int(item["frame"]), -float(item["confidence"])),
    )
    return merged


def consolidate_global_instances(
    instances: list[GlobalInstance],
    image_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> list[GlobalInstance]:
    if len(instances) <= 1:
        return instances

    height, width = image_shape[:2]
    cluster_radius = args.global_cluster_radius_ratio * float(np.hypot(width, height))
    if cluster_radius <= 0:
        return instances

    parents = list(range(len(instances)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    for left in range(len(instances)):
        for right in range(left + 1, len(instances)):
            distance = float(np.linalg.norm(instances[left].global_center - instances[right].global_center))
            if distance <= cluster_radius:
                union(left, right)

    grouped: dict[int, list[GlobalInstance]] = {}
    for index, instance in enumerate(instances):
        grouped.setdefault(find(index), []).append(instance)

    merged_instances = [merge_global_instance_group(group) for group in grouped.values()]
    merged_instances.sort(key=lambda instance: instance.instance_id)
    return merged_instances


def predict_video(model: Any, video_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.perf_counter()
    sampled_frames, actual_last_frame = sample_video_frames(
        video_path, args.max_frames, args.frame_stride, args.video_sampling_backend
    )
    if not sampled_frames:
        raise RuntimeError(f"No readable frames found in: {video_path}")

    predictions = run_batched_predictions(model, sampled_frames, args)
    registration_features = [
        build_registration_features(frame, args.registration_scale) for _frame_index, frame, _result in predictions
    ]

    frames_data: list[dict[str, Any]] = []
    previous_frame: np.ndarray | None = None
    previous_transform: np.ndarray | None = None
    anchor_entries: list[dict[str, Any]] = []
    vis_dir = args.output_dir / "visualizations" / video_path.stem
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, ((frame_index, frame, result), current_features) in enumerate(zip(predictions, registration_features)):
        instances, raw_counts = result_to_instances(result)
        if idx == 0:
            transform = np.eye(3, dtype=np.float32)
            registration_status = {
                "success": True,
                "matches": 0,
                "inliers": 0,
                "message": "reference frame",
                "anchor_success": True,
                "anchor_frame": int(frame_index),
                "anchor_matches": 0,
                "anchor_inliers": 0,
                "used_anchor": True,
                "reused_previous": False,
            }
            registration_failed = False
        else:
            assert previous_frame is not None and previous_transform is not None
            previous_features = registration_features[idx - 1]
            h_curr_to_prev, prev_status = estimate_homography(
                previous_features,
                current_features,
                args.registration_scale,
                args.min_h_inliers,
                args.registration_backend,
            )

            best_anchor_transform: np.ndarray | None = None
            best_anchor_status: dict[str, Any] | None = None
            best_anchor_frame: int | None = None
            should_try_anchor = h_curr_to_prev is None or int(prev_status["inliers"]) < max(60, args.min_h_inliers * 2)
            if should_try_anchor:
                for anchor in anchor_entries[:-1]:
                    if anchor["sample_idx"] != 0 and anchor["sample_idx"] % 3 != 0:
                        continue
                    h_curr_to_anchor, anchor_status = estimate_homography(
                        anchor["features"],
                        current_features,
                        args.registration_scale,
                        args.min_h_inliers,
                        args.registration_backend,
                    )
                    if h_curr_to_anchor is None:
                        continue
                    if best_anchor_status is None or int(anchor_status["inliers"]) > int(best_anchor_status["inliers"]):
                        best_anchor_transform = anchor["transform"] @ h_curr_to_anchor
                        best_anchor_status = dict(anchor_status)
                        best_anchor_frame = int(anchor["frame"])

            used_anchor = False
            if h_curr_to_prev is None and best_anchor_transform is None:
                transform = previous_transform.copy()
                registration_status = dict(prev_status)
                registration_status["anchor_success"] = bool(best_anchor_status is not None)
                registration_status["anchor_frame"] = best_anchor_frame
                registration_status["anchor_matches"] = int(best_anchor_status["matches"]) if best_anchor_status else 0
                registration_status["anchor_inliers"] = int(best_anchor_status["inliers"]) if best_anchor_status else 0
                registration_status["used_anchor"] = False
                registration_status["reused_previous"] = True
                registration_failed = True
            else:
                if h_curr_to_prev is None:
                    assert best_anchor_transform is not None and best_anchor_status is not None
                    transform = best_anchor_transform
                    registration_status = dict(best_anchor_status)
                    used_anchor = True
                else:
                    transform = previous_transform @ h_curr_to_prev
                    registration_status = dict(prev_status)
                    if best_anchor_transform is not None and best_anchor_status is not None:
                        prev_inliers = int(prev_status["inliers"])
                        anchor_inliers = int(best_anchor_status["inliers"])
                        if anchor_inliers >= max(args.min_h_inliers + 5, int(prev_inliers * 0.9)):
                            transform = best_anchor_transform
                            registration_status = dict(best_anchor_status)
                            used_anchor = True
                registration_status["anchor_success"] = bool(best_anchor_status is not None)
                registration_status["anchor_frame"] = best_anchor_frame
                registration_status["anchor_matches"] = int(best_anchor_status["matches"]) if best_anchor_status else 0
                registration_status["anchor_inliers"] = int(best_anchor_status["inliers"]) if best_anchor_status else 0
                registration_status["used_anchor"] = used_anchor
                registration_status["reused_previous"] = False
                registration_failed = False

        frame_data = {
            "frame": frame_index,
            "raw_counts": raw_counts,
            "instances": instances,
            "global_transform": transform,
            "registration_status": registration_status,
            "registration_failed": registration_failed,
        }
        frames_data.append(frame_data)
        previous_frame = frame
        previous_transform = transform
        anchor_entries.append(
            {
                "sample_idx": idx,
                "frame": frame_index,
                "features": current_features,
                "transform": transform,
            }
        )

    valid_instances, registration_status = merge_detections_globally(
        frames_data,
        sampled_frames[0][1].shape,
        args,
    )
    valid_instances = consolidate_global_instances(valid_instances, sampled_frames[0][1].shape, args)

    frame_summaries: dict[str, Any] = {}
    for frame_data, (_frame_index, frame, _result) in zip(frames_data, predictions):
        clean_instances = strip_internal_masks(frame_data["instances"])
        for item, instance_id in zip(clean_instances, frame_data["instance_ids"]):
            item["global_instance_id"] = int(instance_id)
            if "global_center" in item:
                item["global_center"] = item["global_center"]
        frame_name = f'{frame_data["frame"]:03d}'
        frame_summaries[frame_name] = {
            "raw_counts": frame_data["raw_counts"],
            "instances": clean_instances,
            "registration_failed": frame_data["registration_failed"],
            "registration_status": frame_data["registration_status"],
        }
        if args.save_vis:
            vis = draw_instances(frame, frame_data["instances"], frame_data["instance_ids"])
            cv2.imwrite(str(vis_dir / f"{frame_name}.png"), vis)

    elapsed = time.perf_counter() - start_time
    return {
        "mode": "video",
        "source": str(video_path),
        "class_names": CLASS_NAMES,
        "total_counts": counts_from_global_instances(valid_instances),
        "actual_last_frame": actual_last_frame,
        "sampled_frames": [frame_index for frame_index, _frame in sampled_frames],
        "processing_seconds": elapsed,
        "tracker_config": {
            "frame_stride": args.frame_stride,
            "registration_scale": args.registration_scale,
            "registration_backend": args.registration_backend,
            "min_h_inliers": args.min_h_inliers,
            "global_match_radius_ratio": args.global_match_radius_ratio,
            "global_cluster_radius_ratio": args.global_cluster_radius_ratio,
            "min_observations": args.min_observations,
            "quality_conf": args.quality_conf,
            "border_margin": args.border_margin,
            "roi_margin": args.roi_margin,
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "save_observation_log": args.save_observation_log,
        },
        "registration_status": registration_status,
        "global_instances": [instance.to_json() for instance in sorted(valid_instances, key=lambda item: item.instance_id)],
        "frames": frame_summaries,
    }


def count_single_video(
    video_path: Path,
    *,
    model_path: Path | None = None,
    device: str | None = None,
    output_dir: Path | None = None,
    imgsz: int = 960,
    conf: float = 0.25,
    iou: float = 0.7,
    predict_batch: int | None = None,
    max_frames: int = 300,
    frame_stride: int = 10,
    video_sampling_backend: str = "grab",
    registration_scale: float = 0.2,
    registration_backend: str = "orb",
    min_h_inliers: int = 25,
    global_match_radius_ratio: float = 0.03,
    global_cluster_radius_ratio: float = 0.015,
    min_observations: int = 2,
    quality_conf: float = 0.35,
    border_margin: float = 0.03,
    roi_margin: float = 0.05,
    min_mask_area_ratio: float = 0.00005,
    save_observation_log: bool = False,
) -> dict[str, Any]:
    video_path = Path(video_path)
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    args = build_inference_args(
        model_path=model_path,
        output_dir=output_dir,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        predict_batch=predict_batch,
        max_frames=max_frames,
        save_vis=False,
        frame_stride=frame_stride,
        video_sampling_backend=video_sampling_backend,
        registration_scale=registration_scale,
        registration_backend=registration_backend,
        min_h_inliers=min_h_inliers,
        global_match_radius_ratio=global_match_radius_ratio,
        global_cluster_radius_ratio=global_cluster_radius_ratio,
        min_observations=min_observations,
        quality_conf=quality_conf,
        border_margin=border_margin,
        roi_margin=roi_margin,
        min_mask_area_ratio=min_mask_area_ratio,
        save_observation_log=save_observation_log,
    )
    if args.predict_batch is None and args.model_path.suffix.lower() == ".engine":
        args.predict_batch = 1

    model = load_model(args.model_path)
    return predict_video(model, video_path, args)


def count_videos_in_directory(
    data_dir: Path,
    *,
    model_path: Path | None = None,
    device: str | None = None,
    output_dir: Path | None = None,
    imgsz: int = 960,
    conf: float = 0.25,
    iou: float = 0.7,
    predict_batch: int | None = None,
    max_frames: int = 300,
    frame_stride: int = 10,
    video_sampling_backend: str = "grab",
    registration_scale: float = 0.2,
    registration_backend: str = "orb",
    min_h_inliers: int = 25,
    global_match_radius_ratio: float = 0.03,
    global_cluster_radius_ratio: float = 0.015,
    min_observations: int = 2,
    quality_conf: float = 0.35,
    border_margin: float = 0.03,
    roi_margin: float = 0.05,
    min_mask_area_ratio: float = 0.00005,
    save_observation_log: bool = False,
    return_summaries: bool = False,
) -> dict[str, list[int]] | tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {data_dir}")

    video_paths = list_video_files(data_dir)
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {data_dir}")

    args = build_inference_args(
        model_path=model_path,
        output_dir=output_dir,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        predict_batch=predict_batch,
        max_frames=max_frames,
        save_vis=False,
        frame_stride=frame_stride,
        video_sampling_backend=video_sampling_backend,
        registration_scale=registration_scale,
        registration_backend=registration_backend,
        min_h_inliers=min_h_inliers,
        global_match_radius_ratio=global_match_radius_ratio,
        global_cluster_radius_ratio=global_cluster_radius_ratio,
        min_observations=min_observations,
        quality_conf=quality_conf,
        border_margin=border_margin,
        roi_margin=roi_margin,
        min_mask_area_ratio=min_mask_area_ratio,
        save_observation_log=save_observation_log,
    )
    if args.predict_batch is None and args.model_path.suffix.lower() == ".engine":
        args.predict_batch = 1

    model = load_model(args.model_path)
    results: dict[str, list[int]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for video_path in video_paths:
        summary = predict_video(model, video_path, args)
        results[video_path.stem] = [int(value) for value in summary["total_counts"]]
        summaries[video_path.stem] = summary
    if return_summaries:
        return results, summaries
    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.benchmark_video_sampling:
        summary = benchmark_video_sampling(args)
        print(
            json.dumps(
                {
                    "mode": summary["mode"],
                    "global_winner": summary["global_winner"],
                    "aggregate": summary["aggregate"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.source is None:
        raise ValueError("--source is required unless --benchmark_video_sampling is used.")
    if args.predict_batch is None and args.model_path.suffix.lower() == ".engine":
        args.predict_batch = 1
    model = load_model(args.model_path)
    source = args.source

    if source.is_dir():
        image_paths = list_image_files(source)
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {source}")
        vis_dir = args.output_dir / "visualizations"
        if args.save_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)
        per_item: dict[str, dict[str, Any]] = {}
        for image_path in image_paths:
            item, vis = predict_image(model, image_path, args)
            per_item[image_path.name] = item
            if vis is not None:
                cv2.imwrite(str(vis_dir / image_path.name), vis)
        summary = summarize_image_items(per_item, str(source))
    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        item, vis = predict_image(model, source, args)
        if vis is not None:
            vis_dir = args.output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_dir / source.name), vis)
        summary = summarize_image_items({source.name: item}, str(source))
    elif source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        summary = predict_video(model, source, args)
    else:
        raise FileNotFoundError(f"Unsupported source: {source}")

    save_summary(summary, args.output_dir)
    print(json.dumps({"class_names": CLASS_NAMES, "total_counts": summary["total_counts"]}, indent=2))


if __name__ == "__main__":
    main()
