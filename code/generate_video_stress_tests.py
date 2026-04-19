from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class StressPreset:
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    red_shift: float = 0.0
    blue_shift: float = 0.0
    blur_ksize: int = 0
    motion_blur_ksize: int = 0
    noise_sigma: float = 0.0
    jpeg_quality: int | None = None


PRESETS: dict[str, StressPreset] = {
    "bright": StressPreset(brightness=28.0, contrast=1.05, gamma=0.92),
    "dark": StressPreset(brightness=-36.0, contrast=0.95, gamma=1.25),
    "low_contrast": StressPreset(contrast=0.72, brightness=6.0),
    "gaussian_blur": StressPreset(blur_ksize=5),
    "motion_blur": StressPreset(motion_blur_ksize=11),
    "noise": StressPreset(noise_sigma=10.0),
    "jpeg_artifact": StressPreset(jpeg_quality=28),
    "warm": StressPreset(red_shift=18.0, blue_shift=-12.0, saturation=1.05),
    "cool": StressPreset(red_shift=-10.0, blue_shift=16.0, saturation=0.96),
    "combo_mild": StressPreset(
        brightness=-12.0,
        contrast=0.9,
        gamma=1.08,
        saturation=0.94,
        blur_ksize=3,
        jpeg_quality=42,
    ),
    "combo_hard": StressPreset(
        brightness=-28.0,
        contrast=0.82,
        gamma=1.18,
        saturation=0.9,
        red_shift=8.0,
        blue_shift=-8.0,
        motion_blur_ksize=9,
        noise_sigma=8.0,
        jpeg_quality=24,
    ),
}


def parse_args() -> argparse.Namespace:
    repo_hw4 = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate stress-test video variants for robustness evaluation. "
            "Supports single videos or a folder of videos."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Input video file or folder.")
    parser.add_argument("--output_dir", type=Path, default=repo_hw4 / "output" / "stress_tests")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated preset names, or 'all'. Use --list_variants to inspect options.",
    )
    parser.add_argument("--max_frames", type=int, default=0, help="Optional frame cap for quick testing. 0 means all.")
    parser.add_argument("--output_ext", type=str, default=".mp4", help="Output video extension, default .mp4")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--seed", type=int, default=4701, help="Random seed for noise perturbations.")
    parser.add_argument("--list_variants", action="store_true", help="Print available preset names and exit.")
    return parser.parse_args()


def resolve_variants(variants_arg: str) -> list[str]:
    if variants_arg.strip().lower() == "all":
        return list(PRESETS.keys())
    variants = [item.strip() for item in variants_arg.split(",") if item.strip()]
    unknown = [item for item in variants if item not in PRESETS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(PRESETS)}")
    return variants


def list_video_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError(f"Unsupported video file: {source}")
        return [source]
    if source.is_dir():
        videos = sorted([path for path in source.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES])
        if not videos:
            raise FileNotFoundError(f"No video files found in: {source}")
        return videos
    raise FileNotFoundError(f"Source not found: {source}")


def ensure_odd(ksize: int) -> int:
    if ksize <= 1:
        return 0
    return ksize if ksize % 2 == 1 else ksize + 1


def apply_brightness_contrast(frame: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    adjusted = frame.astype(np.float32) * float(contrast) + float(brightness)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    if abs(float(gamma) - 1.0) < 1e-6:
        return frame
    gamma = max(float(gamma), 1e-6)
    table = np.array([((idx / 255.0) ** gamma) * 255.0 for idx in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)


def apply_saturation(frame: np.ndarray, saturation: float) -> np.ndarray:
    if abs(float(saturation) - 1.0) < 1e-6:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(saturation)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_channel_shift(frame: np.ndarray, red_shift: float, blue_shift: float) -> np.ndarray:
    if abs(float(red_shift)) < 1e-6 and abs(float(blue_shift)) < 1e-6:
        return frame
    shifted = frame.astype(np.float32)
    shifted[..., 2] += float(red_shift)
    shifted[..., 0] += float(blue_shift)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_blur(frame: np.ndarray, blur_ksize: int) -> np.ndarray:
    ksize = ensure_odd(blur_ksize)
    if ksize <= 1:
        return frame
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def apply_motion_blur(frame: np.ndarray, motion_blur_ksize: int) -> np.ndarray:
    ksize = ensure_odd(motion_blur_ksize)
    if ksize <= 1:
        return frame
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(frame, -1, kernel)


def apply_noise(frame: np.ndarray, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    if float(noise_sigma) <= 0:
        return frame
    noise = rng.normal(0.0, float(noise_sigma), frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_jpeg_artifact(frame: np.ndarray, jpeg_quality: int | None) -> np.ndarray:
    if jpeg_quality is None:
        return frame
    quality = int(np.clip(jpeg_quality, 5, 100))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return frame
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else frame


def apply_preset(frame: np.ndarray, preset: StressPreset, rng: np.random.Generator) -> np.ndarray:
    output = frame
    output = apply_brightness_contrast(output, preset.brightness, preset.contrast)
    output = apply_gamma(output, preset.gamma)
    output = apply_saturation(output, preset.saturation)
    output = apply_channel_shift(output, preset.red_shift, preset.blue_shift)
    output = apply_blur(output, preset.blur_ksize)
    output = apply_motion_blur(output, preset.motion_blur_ksize)
    output = apply_noise(output, preset.noise_sigma, rng)
    output = apply_jpeg_artifact(output, preset.jpeg_quality)
    return output


def build_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")
    return writer


def process_video(
    video_path: Path,
    output_dir: Path,
    variant_names: Iterable[str],
    output_ext: str,
    max_frames: int,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size for: {video_path}")

    video_output_dir = output_dir / video_path.stem
    outputs = {}
    writers: dict[str, cv2.VideoWriter] = {}
    rngs = {name: np.random.default_rng(seed + idx) for idx, name in enumerate(variant_names)}

    try:
        for name in variant_names:
            output_path = video_output_dir / f"{video_path.stem}__{name}{output_ext}"
            if output_path.exists() and not overwrite:
                outputs[name] = {
                    "preset": asdict(PRESETS[name]),
                    "output_path": str(output_path.resolve()),
                    "skipped_existing": True,
                }
                continue
            writers[name] = build_writer(output_path, fps, width, height)
            outputs[name] = {
                "preset": asdict(PRESETS[name]),
                "output_path": str(output_path.resolve()),
                "skipped_existing": False,
            }

        processed_frames = 0
        while True:
            if max_frames > 0 and processed_frames >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            for name, writer in writers.items():
                transformed = apply_preset(frame, PRESETS[name], rngs[name])
                writer.write(transformed)
            processed_frames += 1
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()

    return {
        "source_path": str(video_path.resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "input_frame_count": frame_count,
        "written_frames": processed_frames,
        "variants": outputs,
    }


def main() -> None:
    args = parse_args()
    if args.list_variants:
        print(json.dumps({"variants": {name: asdict(preset) for name, preset in PRESETS.items()}}, indent=2))
        return

    output_ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"
    variant_names = resolve_variants(args.variants)
    videos = list_video_files(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": str(args.source.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "variants": variant_names,
        "max_frames": int(args.max_frames),
        "videos": {},
    }

    for video_path in videos:
        summary["videos"][video_path.name] = process_video(
            video_path=video_path,
            output_dir=args.output_dir,
            variant_names=variant_names,
            output_ext=output_ext,
            max_frames=args.max_frames,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    manifest_path = args.output_dir / "stress_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "variants": variant_names}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
