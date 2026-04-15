"""Video -> single panorama stitcher.

Algorithm (validated in stitch_debug.py):
  1) Uniformly sample NUM_SAMPLES frames from the video and downscale.
  2) ORB + BFMatcher(Hamming) + Lowe ratio test between each adjacent pair.
  3) RANSAC homography maps frame(k+1) -> frame(k).
  4) Chain the pairwise homographies to get frame(k) -> frame(0).
  5) Compute the enclosing canvas, warp each frame onto it, and composite.

Compositing modes (controlled by --feather):
  feather=0  : hard first-come-first-served paste. Sharpest, but per-frame
               polygonal edges may show as black seams.
  feather>0  : distance-transform feather blend, capped at `feather` pixels.
               Gives a soft ramp of that width at each frame's edge, hiding
               seams. Larger values = smoother transitions.

Usage (CLI):
    python stitch.py                                         # default, feather=0
    python stitch.py --feather 20                            # feather blend all videos
    python stitch.py path/to/video.mov --feather 30          # feather a single video
    python stitch.py path/to/video.mov --frames 15           # use 15 sampled frames

Usage (library):
    from stitch import stitch_video
    pano, dt, n_in, n_chained = stitch_video(
        "test_videos/clip.mov", feather_px=20, num_samples=12)
"""
import argparse
import os
import time
import cv2
import numpy as np

# ---- parameters (tuned from stitch_debug.py observations) ----
NUM_SAMPLES = 10
LONG_SIDE = 960
ORB_FEATURES = 4000
LOWE_RATIO = 0.75
RANSAC_THRESH = 4.0

VIDEO_DIR = "test_videos"
OUT_DIR = "pano_out"


def sample_frames(path, num_samples, long_side):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read frames from {path}")
    idxs = [int(round(i * (total - 1) / (num_samples - 1))) for i in range(num_samples)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        s = long_side / max(h, w)
        if s < 1.0:
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames


def pairwise_homography(img_src, img_dst, orb):
    """Estimate H mapping points in img_src to img_dst."""
    g1 = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None:
        return None
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [m for m_n in knn if len(m_n) == 2
            for m, n in [m_n] if m.distance < LOWE_RATIO * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    return H


def chain_homographies(frames):
    """Return list of H_to_ref[k] mapping frame k -> frame 0."""
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    H_to_ref = [np.eye(3, dtype=np.float64)]
    for k in range(len(frames) - 1):
        H = pairwise_homography(frames[k + 1], frames[k], orb)
        if H is None:
            print(f"  ! pair {k}->{k+1} failed, stopping chain")
            break
        H_to_ref.append(H_to_ref[-1] @ H)
    return H_to_ref


def canvas_bounds(shapes, Hs):
    pts = []
    for (h, w), H in zip(shapes, Hs):
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        pts.append(cv2.perspectiveTransform(c, H))
    pts = np.concatenate(pts, axis=0).reshape(-1, 2)
    mn = np.floor(pts.min(axis=0)).astype(int)
    mx = np.ceil(pts.max(axis=0)).astype(int)
    return mn, mx


def _canvas_layout(frames, H_to_ref):
    n = len(H_to_ref)
    shapes = [f.shape[:2] for f in frames[:n]]
    mn, mx = canvas_bounds(shapes, H_to_ref)
    offset = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]], dtype=np.float64)
    cw, ch = int(mx[0] - mn[0]), int(mx[1] - mn[1])
    return n, offset, cw, ch


def _hard_paste(frames, H_to_ref):
    """First-come-first-served paste: sharpest, leaves visible frame edges."""
    n, offset, cw, ch = _canvas_layout(frames, H_to_ref)
    canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
    occupied = np.zeros((ch, cw), dtype=bool)
    for k in range(n):
        Hk = offset @ H_to_ref[k]
        warped = cv2.warpPerspective(frames[k], Hk, (cw, ch))
        mask = warped.sum(axis=2) > 0
        write = mask & ~occupied
        canvas[write] = warped[write]
        occupied |= mask
    return canvas


def _feather_blend(frames, H_to_ref, feather_px):
    """Distance-transform feather blend with ramp width = feather_px pixels.

    Each frame's warp mask is eroded by `feather_px` to drop the dark
    bilinear halo, then a distance transform capped at `feather_px` is used
    as the blend weight -- giving a soft ramp of `feather_px` width at each
    frame's edge and a flat plateau in its interior.
    """
    n, offset, cw, ch = _canvas_layout(frames, H_to_ref)
    acc = np.zeros((ch, cw, 3), dtype=np.float32)
    wsum = np.zeros((ch, cw), dtype=np.float32)
    erode_px = max(1, feather_px // 2)
    kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)

    for k in range(n):
        Hk = offset @ H_to_ref[k]
        warped = cv2.warpPerspective(frames[k], Hk, (cw, ch))
        h, w = frames[k].shape[:2]
        ones = np.full((h, w), 255, dtype=np.uint8)
        mask = cv2.warpPerspective(ones, Hk, (cw, ch))
        mask = (mask > 250).astype(np.uint8)
        mask = cv2.erode(mask, kernel)
        weight = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        np.minimum(weight, float(feather_px), out=weight)
        acc += warped.astype(np.float32) * weight[..., None]
        wsum += weight

    out = np.full((ch, cw, 3), 255, dtype=np.uint8)
    valid = wsum > 1e-6
    out[valid] = (acc[valid] / wsum[valid, None]).clip(0, 255).astype(np.uint8)
    return out


def build_panorama(frames, H_to_ref, feather_px=0):
    """Composite the chained frames into a single panorama.

    feather_px = 0  -> hard paste (sharp, leaves edge seams)
    feather_px > 0  -> feather blend with ramp width `feather_px`
    """
    if feather_px <= 0:
        return _hard_paste(frames, H_to_ref)
    return _feather_blend(frames, H_to_ref, int(feather_px))


def stitch_video(path, feather_px=0, num_samples=NUM_SAMPLES):
    """Stitch a video into a panorama.

    Args:
        path: path to the input video file.
        feather_px: edge feather width in pixels. 0 = hard paste (sharpest),
            >0 = distance-transform feather blend of that ramp width.
        num_samples: number of frames to uniformly sample from the video and
            feed into the stitching chain.

    Returns:
        (pano, dt, n_sampled, n_chained): the panorama image, elapsed seconds,
        how many frames were actually sampled, and how many made it into the
        homography chain before it was cut (equal on success).
    """
    t0 = time.time()
    frames = sample_frames(path, num_samples, LONG_SIDE)
    H_chain = chain_homographies(frames)
    pano = build_panorama(frames, H_chain, feather_px=feather_px)
    dt = time.time() - t0
    return pano, dt, len(frames), len(H_chain)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", default=None,
                    help="single video file; if omitted, all videos in test_videos/")
    ap.add_argument("--feather", type=int, default=0,
                    help="edge feather width in pixels (0 = no blur, default)")
    ap.add_argument("--frames", type=int, default=NUM_SAMPLES,
                    help=f"number of frames uniformly sampled from the video "
                         f"(default {NUM_SAMPLES})")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    if args.video:
        paths = [args.video]
    else:
        paths = [os.path.join(VIDEO_DIR, f) for f in sorted(os.listdir(VIDEO_DIR))
                 if f.lower().endswith((".mov", ".mp4"))]
    os.makedirs(args.out_dir, exist_ok=True)
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"\n=== {name} ===")
        pano, dt, n_in, n_chained = stitch_video(
            p, feather_px=args.feather, num_samples=args.frames)
        out = os.path.join(args.out_dir, f"{name}_pano.jpg")
        cv2.imwrite(out, pano)
        print(f"  chained {n_chained}/{n_in} frames, canvas={pano.shape[1]}x{pano.shape[0]}, "
              f"feather={args.feather}, frames={args.frames}, {dt:.2f}s")
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
