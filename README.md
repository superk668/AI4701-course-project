# AI4701 HW4 Counting Package

This branch packages the current best-performing screw-counting pipeline for
video-level inference.

## Repository Layout

```text
AI4701-course-project/
├── code/
│   ├── count_instances.py
│   ├── generate_video_stress_tests.py
│   ├── train_counting_seg.py
│   └── visualize_homography_alignment.py
├── model/
│   └── best.pt
├── requirements.txt
└── README.md
```

## What This Package Does

The core counting logic lives in `code/count_instances.py`.

It supports:

- YOLOv8 segmentation inference on sampled video frames
- homography-based global coordinate alignment
- cross-frame deduplication
- final five-class screw counting

It does **not** write `result.npy`, `time.txt`, or mask images by itself.
Those are expected to be handled by an external `run.py` or other caller.

## Main Python API

The two recommended interfaces are:

### 1. Count all videos in a folder

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/AI4701-course-project")
sys.path.insert(0, str(repo_root / "code"))

from count_instances import count_videos_in_directory

result_dict = count_videos_in_directory(
    data_dir=repo_root / "test_videos",
    model_path=repo_root / "model" / "best.pt",
    device="0",          # or "cpu"
    max_frames=300,
    frame_stride=10,
)

print(result_dict)
```

Return format:

```python
{
    "IMG_2374": [14, 7, 6, 22, 3],
    "IMG_2375": [15, 9, 9, 17, 4],
    "IMG_2376": [15, 5, 6, 12, 4],
}
```

- key: video filename without suffix
- value: length-5 list in the fixed order
  `[Type1, Type2, Type3, Type4, Type5]`

If detailed per-video summaries are needed:

```python
result_dict, summaries = count_videos_in_directory(
    data_dir=repo_root / "test_videos",
    model_path=repo_root / "model" / "best.pt",
    return_summaries=True,
)
```

### 2. Count a single video

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/AI4701-course-project")
sys.path.insert(0, str(repo_root / "code"))

from count_instances import count_single_video

summary = count_single_video(
    video_path=repo_root / "test_videos" / "IMG_2374.MOV",
    model_path=repo_root / "model" / "best.pt",
    device="0",
)

counts = summary["total_counts"]
print(counts)
```

## Common Parameters

Useful parameters exposed by the API:

- `model_path`: weight path, typically `model/best.pt`
- `device`: `"0"` for GPU 0, or `"cpu"`
- `max_frames`: max number of frames to process, default `300`
- `frame_stride`: sampled-frame stride, default `10`
- `imgsz`: YOLO inference image size, default `960`
- `conf`: confidence threshold, default `0.25`
- `iou`: NMS IoU threshold, default `0.7`

## How External `run.py` Should Save `result.npy`

This package only returns Python objects. A caller can save the final result as:

```python
import numpy as np
from pathlib import Path

result_dict = {
    "IMG_2374": [14, 7, 6, 22, 3],
    "IMG_2375": [15, 9, 9, 17, 4],
}

output_path = Path("result.npy")
with output_path.open("wb") as handle:
    np.save(handle, result_dict, allow_pickle=True)
```

The grader should then be able to load it using:

```python
result = np.load("result.npy", allow_pickle=True).item()
```

## Notes

- `model/best.pt` is the current best-performing model kept for submission use.
- `code/train_counting_seg.py` is for training and is not required during test-time inference.
- `code/generate_video_stress_tests.py` and
  `code/visualize_homography_alignment.py` are auxiliary scripts and are included
  because the request was to package all Python files currently under `HW4/code`.
