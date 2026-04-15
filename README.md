# AI4701-course-project

## Video panorama stitcher (`stitch.py`)

Samples frames from a video, chains ORB+RANSAC homographies, and composites
them into a single panorama.

### Command line

```bash
# stitch every video in test_videos/ with default settings (feather=0, frames=10)
python stitch.py

# feather-blend all videos with a 20 px ramp
python stitch.py --feather 20

# stitch a single video with a custom feather width and frame count
python stitch.py path/to/video.mov --feather 30 --frames 15
```

Flags:

- `video` (positional, optional): single video file. If omitted, every
  `.mov`/`.mp4` under `test_videos/` is processed.
- `--feather INT`: edge feather width in pixels. `0` = hard paste (sharpest,
  may show polygonal seams); `>0` = distance-transform feather blend of that
  ramp width. Default `0`.
- `--frames INT`: number of frames uniformly sampled from the video and fed
  into the stitching chain. Default `10`.
- `--out_dir DIR`: output directory for `{name}_pano.jpg`. Default `pano_out`.

### Library usage

`stitch_video` is the single entry point:

```python
from stitch import stitch_video

pano, dt, n_sampled, n_chained = stitch_video(
    "test_videos/clip.mov",
    feather_px=20,     # 0 = hard paste, >0 = feather-blend ramp width
    num_samples=12,    # how many frames to extract and use
)

import cv2
cv2.imwrite("clip_pano.jpg", pano)
```

Returns `(pano, elapsed_seconds, n_sampled, n_chained)`. `n_chained` equals
`n_sampled` on success; it will be smaller if the homography chain was cut
early because a pair failed to match.
