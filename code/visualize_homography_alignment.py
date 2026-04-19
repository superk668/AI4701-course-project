from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from count_instances import (
    build_registration_features,
    estimate_homography,
    sample_video_frames,
)


def parse_args() -> argparse.Namespace:
    repo_hw4 = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Visualize sampled video frames after mapping them into the first keyframe's "
            "global coordinate system with the same homography chaining logic used by count_instances.py."
        )
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        default=repo_hw4 / "vedio_exp" / "vedio_exp" / "IMG_2374.MOV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=repo_hw4 / "output" / "homography_alignment" / "video1",
    )
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--video_sampling_backend", choices=["sequential", "grab", "seek"], default="grab")
    parser.add_argument("--registration_scale", type=float, default=0.2)
    parser.add_argument("--registration_backend", choices=["auto", "lk", "orb"], default="orb")
    parser.add_argument("--min_h_inliers", type=int, default=25)
    parser.add_argument(
        "--contact_indices",
        type=str,
        default="1,6,11,16,21,26,31",
        help="1-based sampled-frame positions to place on the contact sheet.",
    )
    return parser.parse_args()


def draw_status_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    line_height = 32
    pad_x = 16
    pad_y = 18
    box_height = pad_y * 2 + line_height * len(lines)
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 900), box_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        y = pad_y + int((index + 1) * line_height)
        cv2.putText(output, line, (pad_x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def make_triptych(reference: np.ndarray, current: np.ndarray, warped: np.ndarray, status: dict[str, Any]) -> np.ndarray:
    overlay = cv2.addWeighted(reference, 0.5, warped, 0.5, 0.0)
    top = np.hstack([reference, current, warped])
    bottom = np.hstack(
        [
            draw_status_text(reference, ["Reference keyframe"]),
            draw_status_text(
                current,
                [
                    f'Frame {int(status["frame"]):03d}',
                    f'Source view',
                ],
            ),
            draw_status_text(
                overlay,
                [
                    f'used_anchor={bool(status["used_anchor"])}',
                    f'inliers={int(status["inliers"])} anchor={status["anchor_frame"]}',
                    f'msg={status["message"]}',
                ],
            ),
        ]
    )
    return np.vstack([top, bottom])


def image_corners(width: int, height: int) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)


def warp_full_image(frame: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    height, width = frame.shape[:2]
    corners = image_corners(width, height)
    warped_corners = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
    min_x = float(np.floor(warped_corners[:, 0].min()))
    min_y = float(np.floor(warped_corners[:, 1].min()))
    max_x = float(np.ceil(warped_corners[:, 0].max()))
    max_y = float(np.ceil(warped_corners[:, 1].max()))

    translation = np.array(
        [
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    output_width = max(1, int(max_x - min_x + 1))
    output_height = max(1, int(max_y - min_y + 1))
    full_transform = translation @ transform
    warped_full = cv2.warpPerspective(frame, full_transform, (output_width, output_height))
    bounds = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": float(output_width),
        "height": float(output_height),
    }
    return warped_full, full_transform, bounds


def make_full_overlay(
    reference: np.ndarray,
    current: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    ref_height, ref_width = reference.shape[:2]
    curr_height, curr_width = current.shape[:2]
    ref_corners = image_corners(ref_width, ref_height)
    curr_corners = cv2.perspectiveTransform(image_corners(curr_width, curr_height), transform)
    all_corners = np.vstack([ref_corners.reshape(-1, 2), curr_corners.reshape(-1, 2)])

    min_x = float(np.floor(all_corners[:, 0].min()))
    min_y = float(np.floor(all_corners[:, 1].min()))
    max_x = float(np.ceil(all_corners[:, 0].max()))
    max_y = float(np.ceil(all_corners[:, 1].max()))

    translation = np.array(
        [
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    canvas_width = max(1, int(max_x - min_x + 1))
    canvas_height = max(1, int(max_y - min_y + 1))

    ref_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    offset_x = int(round(-min_x))
    offset_y = int(round(-min_y))
    ref_canvas[offset_y : offset_y + ref_height, offset_x : offset_x + ref_width] = reference
    warped_canvas = cv2.warpPerspective(current, translation @ transform, (canvas_width, canvas_height))
    overlay = cv2.addWeighted(ref_canvas, 0.5, warped_canvas, 0.5, 0.0)
    bounds = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": float(canvas_width),
        "height": float(canvas_height),
    }
    return overlay, bounds


def save_contact_sheet(selected_paths: list[Path], output_path: Path) -> None:
    if not selected_paths:
        return
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in selected_paths]
    images = [image for image in images if image is not None]
    if not images:
        return

    tile_width = 640
    resized = []
    for image in images:
        scale = tile_width / image.shape[1]
        tile_height = max(1, int(round(image.shape[0] * scale)))
        resized.append(cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA))

    rows: list[np.ndarray] = []
    for start in range(0, len(resized), 2):
        row_images = resized[start : start + 2]
        max_height = max(image.shape[0] for image in row_images)
        padded = []
        for image in row_images:
            if image.shape[0] < max_height:
                pad = np.zeros((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
                padded.append(np.vstack([image, pad]))
            else:
                padded.append(image)
        if len(padded) == 1:
            padded.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(padded))
    contact = np.vstack(rows)
    cv2.imwrite(str(output_path), contact)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sampled_frames, actual_last_frame = sample_video_frames(
        args.video_path,
        args.max_frames,
        args.frame_stride,
        args.video_sampling_backend,
    )
    if not sampled_frames:
        raise RuntimeError(f"No sampled frames from {args.video_path}")

    reference_frame = sampled_frames[0][1]
    height, width = reference_frame.shape[:2]
    registration_features = [
        build_registration_features(frame, args.registration_scale) for _frame_index, frame in sampled_frames
    ]

    frame_metadata: list[dict[str, Any]] = []
    previous_transform: np.ndarray | None = None
    anchor_entries: list[dict[str, Any]] = []
    selected_positions = {
        max(1, int(item.strip()))
        for item in args.contact_indices.split(",")
        if item.strip()
    }
    contact_paths: list[Path] = []

    for idx, ((frame_index, frame), current_features) in enumerate(zip(sampled_frames, registration_features)):
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
                "frame": int(frame_index),
            }
        else:
            assert previous_transform is not None
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
                registration_status["reused_previous"] = True
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

                registration_status["reused_previous"] = False

            registration_status["anchor_success"] = bool(best_anchor_status is not None)
            registration_status["anchor_frame"] = best_anchor_frame
            registration_status["anchor_matches"] = int(best_anchor_status["matches"]) if best_anchor_status else 0
            registration_status["anchor_inliers"] = int(best_anchor_status["inliers"]) if best_anchor_status else 0
            registration_status["used_anchor"] = used_anchor
            registration_status["frame"] = int(frame_index)

        warped = cv2.warpPerspective(frame, transform, (width, height))
        warped_full, full_transform, warped_full_bounds = warp_full_image(frame, transform)
        overlay = cv2.addWeighted(reference_frame, 0.5, warped, 0.5, 0.0)
        overlay_full, overlay_full_bounds = make_full_overlay(reference_frame, frame, transform)

        frame_stem = f"{idx + 1:02d}_frame_{frame_index:03d}"
        current_path = args.output_dir / f"{frame_stem}_current.png"
        warped_path = args.output_dir / f"{frame_stem}_warped.png"
        warped_full_path = args.output_dir / f"{frame_stem}_warped_full.png"
        overlay_path = args.output_dir / f"{frame_stem}_overlay.png"
        overlay_full_path = args.output_dir / f"{frame_stem}_overlay_full.png"
        triptych_path = args.output_dir / f"{frame_stem}_triptych.png"
        cv2.imwrite(str(current_path), frame)
        cv2.imwrite(str(warped_path), warped)
        cv2.imwrite(str(warped_full_path), warped_full)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(overlay_full_path), overlay_full)
        cv2.imwrite(str(triptych_path), make_triptych(reference_frame, frame, warped, registration_status))

        if (idx + 1) in selected_positions:
            contact_paths.append(triptych_path)

        frame_metadata.append(
            {
                "sample_position": idx + 1,
                "frame": int(frame_index),
                "current_path": str(current_path),
                "warped_path": str(warped_path),
                "warped_full_path": str(warped_full_path),
                "overlay_path": str(overlay_path),
                "overlay_full_path": str(overlay_full_path),
                "triptych_path": str(triptych_path),
                "warped_full_transform": [[float(value) for value in row] for row in full_transform.tolist()],
                "warped_full_bounds": warped_full_bounds,
                "overlay_full_bounds": overlay_full_bounds,
                "registration_status": registration_status,
            }
        )
        previous_transform = transform
        anchor_entries.append(
            {
                "sample_idx": idx,
                "frame": frame_index,
                "features": current_features,
                "transform": transform,
            }
        )

    save_contact_sheet(contact_paths, args.output_dir / "contact_sheet.png")

    summary = {
        "video_path": str(args.video_path),
        "actual_last_frame": int(actual_last_frame),
        "sampled_frames": [int(frame_index) for frame_index, _frame in sampled_frames],
        "reference_frame": int(sampled_frames[0][0]),
        "output_dir": str(args.output_dir),
        "frames": frame_metadata,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "sample_count": len(sampled_frames)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
