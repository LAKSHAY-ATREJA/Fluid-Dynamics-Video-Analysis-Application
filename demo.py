"""
demo.py — Headless demonstration of the core algorithms.

This script generates a synthetic fluid-flow video (a sinusoidal wave pattern
with added Gaussian noise), runs the optical flow and SfS depth estimation on
it without any GUI interaction, and saves three output files:

  demo_input.mp4             — the synthetic video
  demo_velocity_heatmap.png  — per-pixel velocity heatmap (JET colormap)
  demo_velocity_profile.png  — horizontal velocity profile at mid-frame

No real video file or Qt installation is required to run this demo.

Usage
-----
    python demo.py

All output files are written to the current working directory.
"""

import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure the project root is on the path when running from a subdirectory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import apply_clahe, apply_gamma_correction
from SFSImplementation import run_sfs_on_frame, forward_gradients


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_PATH = "demo_input.mp4"
OUT_HEATMAP = "demo_velocity_heatmap.png"
OUT_PROFILE = "demo_velocity_profile.png"

FRAME_W = 640
FRAME_H = 480
NUM_FRAMES = 60          # frames to generate and process
FPS = 25.0
MPP = 0.039              # metres per pixel (representative value)
GAMMA = 2.0              # gamma correction applied to the synthetic frames
SFS_ITERS = 15           # keep low for fast demo execution


# ---------------------------------------------------------------------------
# Synthetic video generation
# ---------------------------------------------------------------------------

def make_synthetic_frame(t: float, h: int, w: int) -> np.ndarray:
    """
    Generate a single BGR frame simulating a flowing-water surface.

    The pattern is a horizontal sine wave that translates to the right over
    time (controlled by *t* in seconds), with additive Gaussian noise and a
    second-harmonic component to mimic turbulence.
    """
    xs = np.linspace(0, 4 * np.pi, w)
    ys = np.linspace(0, 2 * np.pi, h)
    X, Y = np.meshgrid(xs, ys)

    wave = (
        0.5 * np.sin(X - 3.0 * t)
        + 0.25 * np.sin(2 * X - 5.0 * t + Y * 0.3)
        + 0.1 * np.random.randn(h, w)
    )
    # Normalise to [0, 255]
    wave_min, wave_max = wave.min(), wave.max()
    gray = ((wave - wave_min) / (wave_max - wave_min + 1e-8) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def generate_synthetic_video(path: str, num_frames: int, fps: float, h: int, w: int):
    """Write a synthetic fluid-flow video to *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Cannot open VideoWriter for: {path}")

    rng_seed = 42
    np.random.seed(rng_seed)
    for i in range(num_frames):
        t = i / fps
        frame = make_synthetic_frame(t, h, w)
        writer.write(frame)

    writer.release()
    print(f"Synthetic video written: {path}  ({num_frames} frames at {fps} fps)")


# ---------------------------------------------------------------------------
# Headless analysis
# ---------------------------------------------------------------------------

def run_demo():
    print("=== Fluid Dynamics Video Analysis — Headless Demo ===\n")

    # 1. Generate synthetic input video
    print("[1/4] Generating synthetic fluid-flow video...")
    generate_synthetic_video(VIDEO_PATH, NUM_FRAMES, FPS, FRAME_H, FRAME_W)

    # 2. Open video and process frames
    print("[2/4] Running optical flow and SfS depth estimation...")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {VIDEO_PATH}")

    # Use the full frame as the single ROI
    roi = (0, 0, FRAME_W, FRAME_H)

    ok, prev_raw = cap.read()
    if not ok:
        cap.release()
        raise IOError("Cannot read first frame")

    prev_gray = cv2.cvtColor(prev_raw, cv2.COLOR_BGR2GRAY)
    prev_gray = apply_clahe(prev_gray)
    prev_gray = apply_gamma_correction(prev_gray, GAMMA)

    velocity_sum = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
    frame_count = 0
    np.random.seed(42)  # reproducible noise in subsequent frames

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = apply_clahe(gray)
        gray_corrected = apply_gamma_correction(gray, GAMMA)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray_corrected, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # SfS refinement
        sfs_in = gray_corrected.astype(np.float32) / 255.0
        z = run_sfs_on_frame(sfs_in, iters=SFS_ITERS)
        zx, zy = forward_gradients(z)
        downslope = np.stack([-zx, -zy], axis=-1)
        downslope /= np.linalg.norm(downslope, axis=-1, keepdims=True) + 1e-8

        flow_unit = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-8)
        alignment = np.sum(flow_unit * downslope, axis=-1)

        raw_weights = np.clip((alignment + 1) / 2, 0, 1)
        raw_weights[alignment < 0] *= 0.5   # penalise anti-downslope flow
        influence = 0.7
        weights = 1.0 - influence * (1.0 - raw_weights)
        flow_refined = flow * weights[..., None]

        fx, fy = flow_refined[..., 0], flow_refined[..., 1]
        mag, _ = cv2.cartToPolar(fx, fy, angleInDegrees=True)
        velocity = mag * FPS * MPP  # pixel displacement -> m/s

        velocity_sum += velocity.astype(np.float64)
        frame_count += 1
        prev_gray = gray_corrected.copy()

        if frame_count % 10 == 0:
            print(f"  Processed frame {frame_count}/{NUM_FRAMES - 1}")

    cap.release()
    print(f"  Processed {frame_count} frames total.")

    if frame_count == 0:
        print("No frames processed. Exiting.")
        return

    avg_velocity = (velocity_sum / frame_count).astype(np.float32)

    # 3. Velocity heatmap
    print("[3/4] Saving velocity heatmap...")
    vmax = float(np.percentile(avg_velocity, 99))
    capped = np.clip(avg_velocity, 0, vmax)
    normed = cv2.normalize(capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(normed, cv2.COLORMAP_JET)

    # Overlay a simple colorbar strip on the right edge
    bar_w = 30
    bar_grad = np.linspace(255, 0, FRAME_H).astype(np.uint8).reshape(-1, 1)
    bar_color = cv2.applyColorMap(bar_grad, cv2.COLORMAP_JET)
    bar_color = np.repeat(bar_color, bar_w, axis=1)
    out_img = np.hstack([heatmap_bgr, bar_color])
    # Label the bar
    cv2.putText(out_img, f"{vmax:.2f} m/s", (FRAME_W + 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out_img, "0.00 m/s", (FRAME_W + 2, FRAME_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(OUT_HEATMAP, out_img)
    print(f"  Saved: {OUT_HEATMAP}")

    # 4. Horizontal velocity profile at mid-frame row
    print("[4/4] Saving velocity profile plot...")
    mid_row = FRAME_H // 2
    profile = avg_velocity[mid_row, :]
    xs_m = np.arange(FRAME_W) * MPP

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs_m, profile, color="steelblue", linewidth=1.8, label="Measured velocity")
    ax.fill_between(xs_m, profile, alpha=0.25, color="steelblue")
    ax.set_xlabel("Cross-stream position (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(
        f"Horizontal velocity profile at mid-frame row (y = {mid_row * MPP:.2f} m)\n"
        f"Synthetic demo video — {frame_count} frames averaged at {FPS} fps"
    )
    ax.set_xlim(0, xs_m[-1])
    ax.set_ylim(0, max(float(profile.max()) * 1.2, 0.01))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PROFILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_PROFILE}")

    # Summary
    print(f"\n=== Demo complete ===")
    print(f"  Mean velocity (whole frame):  {float(avg_velocity.mean()):.4f} m/s")
    print(f"  99th percentile velocity:     {vmax:.4f} m/s")
    print(f"\nOutput files:")
    print(f"  {VIDEO_PATH}")
    print(f"  {OUT_HEATMAP}")
    print(f"  {OUT_PROFILE}")
    print(
        "\nTo run the full interactive application on a real video:\n"
        "  python main.py --video /path/to/video.MOV\n"
    )


if __name__ == "__main__":
    run_demo()
