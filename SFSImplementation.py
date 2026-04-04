"""
SFSImplementation.py — Standalone Shape-from-Shading depth estimation pipeline.

Given a video of flowing water, this module:
  1. Asks the user to select two regions of interest and tune gamma per region.
  2. Iteratively solves for a per-frame depth map using the photometric
     Shape-from-Shading (SfS) gradient-descent formulation.
  3. Converts the relative SfS depth to physically scaled water depth using
     four user-selected calibration points.
  4. Writes two heatmap videos (relative SfS depth and physical depth) and
     three diagnostic images (3D surface, contour map, depth-profile comparison).

The core SfS functions ``forward_gradients`` and ``run_sfs_on_frame`` are also
imported by ``main.py`` to refine optical-flow direction estimates.

Usage
-----
    python SFSImplementation.py [--video PATH] [--max-frames N]

Defaults to the project's original input video name when no argument is given.
"""

import argparse
import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
import pandas as pd

from preprocessing import apply_gamma_correction


# ---------------------------------------------------------------------------
# Default output paths
# ---------------------------------------------------------------------------
OUTPUT_SFS_HEATMAP = "sfs_depth_heatmap.mp4"
OUTPUT_PHYSICAL_HEATMAP = "physical_depth_heatmap.mp4"
OUT_CONTOUR_MAP = "physical_depth_contour_map.png"
OUT_3D_DEPTH_PNG = "average_physical_depth_map.png"
OUT_DEPTH_PROFILE = "depth_profile_physical_vs_sfs.png"

# ---------------------------------------------------------------------------
# Photometric model
# ---------------------------------------------------------------------------
rho = 1.0
LIGHT_DIRECTION = np.array([0.2, 0.1, 0.97])  # assumed light-source direction
s = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)

# ---------------------------------------------------------------------------
# Optimisation parameters
# ---------------------------------------------------------------------------
SMOOTHNESS_LAMBDA = 1.0
LEARNING_RATE_ALPHA = 0.05
SFS_ITERATIONS = 400
GAUSSIAN_KERNEL = (7, 7)
SPECULAR_THRESHOLD = 245  # pixels brighter than this are masked as specular

# ---------------------------------------------------------------------------
# Dam reference elevations (metres) — used to convert relative SfS to physical
# ---------------------------------------------------------------------------
Z_TOP_M = 93.0
Z_TURB_M = 84.0
Z_END_M = 75.0

# ---------------------------------------------------------------------------
# Known real-world distances for spatial calibration
# ---------------------------------------------------------------------------
DAM_WIDTH_METRES = 12.25
DAM_HEIGHT_METRES = 21.0  # vertical distance to point of interest

# ---------------------------------------------------------------------------
# Temporal initialisation
# ---------------------------------------------------------------------------
use_temporal_init = True
frame_stride = 1  # process every Nth frame


# ===========================================================================
# Core SfS mathematics
# ===========================================================================

def forward_gradients(z):
    """
    Compute forward-difference spatial gradients of depth map *z*.

    Returns ``(zx, zy)`` — horizontal and vertical gradient arrays of the
    same shape as *z*.  Boundary columns/rows are zero-padded.
    """
    zx = np.zeros_like(z)
    zy = np.zeros_like(z)
    zx[:, :-1] = z[:, 1:] - z[:, :-1]
    zy[:-1, :] = z[1:, :] - z[:-1, :]
    return zx, zy


def laplacian(z):
    """
    Discrete Laplacian of *z* using zero-Dirichlet boundary conditions.
    Used as the smoothness regulariser in the SfS gradient descent.
    """
    lap = -4.0 * z.copy()
    lap += np.pad(z[1:, :],  ((0, 1), (0, 0)), mode="constant")
    lap += np.pad(z[:-1, :], ((1, 0), (0, 0)), mode="constant")
    lap += np.pad(z[:, 1:],  ((0, 0), (0, 1)), mode="constant")
    lap += np.pad(z[:, :-1], ((0, 0), (1, 0)), mode="constant")
    return lap


def run_sfs_on_frame(
    I,
    z_init=None,
    s=s,
    rho=rho,
    lam=SMOOTHNESS_LAMBDA,
    alpha=LEARNING_RATE_ALPHA,
    iters=200,
    mask=None,
):
    """
    Perform Shape-from-Shading on a single normalised intensity image *I*.

    Iteratively refines the depth map ``z`` to minimise the photometric
    error ``(R(z) - I)^2`` subject to a Laplacian smoothness penalty, where
    ``R(z) = rho * (-sx*zx - sy*zy + sz) / sqrt(1 + zx^2 + zy^2)`` is the
    Lambertian reflectance model evaluated at the surface normals derived from
    the gradient of *z*.

    Parameters
    ----------
    I : ndarray, shape (H, W), float32 in [0, 1]
        Observed intensity image.
    z_init : ndarray or None
        Initial depth estimate.  Pass ``None`` to start from a flat surface.
    s : array-like, shape (3,)
        Unit light-source direction vector ``[sx, sy, sz]``.
    rho : float
        Surface albedo.
    lam : float
        Smoothness regularisation weight.
    alpha : float
        Gradient-descent step size.
    iters : int
        Number of gradient-descent iterations.
    mask : ndarray or None
        Binary mask (1 = valid pixel, 0 = ignore).

    Returns
    -------
    z : ndarray, shape (H, W), float32
        Estimated relative depth map.
    """
    H, W = I.shape
    if mask is None:
        mask = np.ones((H, W), dtype=np.float32)

    z = (
        np.zeros((H, W), dtype=np.float32)
        if z_init is None
        else z_init.astype(np.float32).copy()
    )
    z[mask == 0] = 0.0

    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    eps = 1e-8

    for _ in range(iters):
        zx, zy = forward_gradients(z)
        den_sq = 1.0 + zx * zx + zy * zy
        den = np.sqrt(den_sq)
        N = rho * (-sx * zx - sy * zy + sz)
        R = N / (den + eps)
        res = (R - I) * mask

        den_pow3 = den_sq * den + eps
        dR_dzx = (-rho * sx * den_sq - N * zx) / (den_pow3 + eps)
        dR_dzy = (-rho * sy * den_sq - N * zy) / (den_pow3 + eps)

        p_x = res * dR_dzx
        p_y = res * dR_dzy

        p_x_left = np.zeros_like(p_x)
        p_x_left[:, 1:] = p_x[:, :-1]
        div_x = p_x - p_x_left

        p_y_up = np.zeros_like(p_y)
        p_y_up[1:, :] = p_y[:-1, :]
        div_y = p_y - p_y_up

        grad_data = -(div_x + div_y)
        grad_smooth = -lam * (laplacian(z) * mask)
        grad_total = (grad_data + grad_smooth) * mask

        z -= alpha * grad_total
        z[mask == 0] = 0.0

    return z


# ===========================================================================
# Physical depth conversion
# ===========================================================================

def convert_sfs_to_physical_depth(z_sfs, z_top_m=Z_TOP_M, z_turb_m=Z_TURB_M, z_end_m=Z_END_M):
    """
    Map relative SfS depth *z_sfs* to physical water depth in metres.

    The SfS output is normalised to [0, 1] then linearly mapped onto the
    known elevation range ``[z_turb_m, z_top_m]``.  Water depth is the
    distance above ``z_end_m``; negative values are clamped to zero.

    Returns ``(h_water, mask)`` where *mask* marks pixels above ``z_end_m``.
    """
    z_range = z_sfs.max() - z_sfs.min()
    z_norm = (z_sfs - z_sfs.min()) / (z_range + 1e-8)
    z_physical = z_norm * (z_top_m - z_turb_m) + z_turb_m
    h_water = z_physical - z_end_m
    h_water[h_water < 0] = 0.0
    mask = z_physical >= z_end_m
    return h_water, mask


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def create_colorbar(height, colormap, min_val, max_val, width=50, label="Depth (m)"):
    """
    Build a vertical OpenCV colorbar image of size ``(height, width + 60)``
    with five labelled tick marks and a rotated axis label.
    """
    height = max(height, 10)
    gradient = np.linspace(0, 255, height).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(gradient, colormap)
    colorbar = cv2.resize(colorbar, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (255, 255, 255)

    tick_positions = np.linspace(0, height - 1, 5).astype(int)
    tick_values = np.linspace(max_val, min_val, 5)  # top = max

    for pos, val in zip(tick_positions, tick_values):
        cv2.putText(
            colorbar, f"{val:.2f} m", (5, int(pos) + 5),
            font, font_scale, text_color, 1, cv2.LINE_AA,
        )

    label_width = 60
    temp_img = np.zeros((label_width, height, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
    text_x = max(0, (height - text_size[0]) // 2)
    text_y = max(text_size[1], (label_width + text_size[1]) // 2)
    cv2.putText(temp_img, label, (text_x, text_y), font, 0.6, text_color, 1, cv2.LINE_AA)
    label_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    label_img = cv2.resize(label_img, (label_width, height))

    return np.hstack([label_img, colorbar])


def plot_average_physical_depth(depth_map, save_path, x_scale, y_scale, origin_x, origin_y):
    """Render a 3D surface of the average physical depth and save it to *save_path*."""
    H, W = depth_map.shape
    X = (np.arange(W) - origin_x) * x_scale
    Y = (np.arange(H) - origin_y) * (-y_scale)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, depth_map, cmap="inferno", linewidth=0, antialiased=True)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("Average Physical Depth Map (Metre Calibrated)")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Depth (m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D depth map to: {save_path}")


def plot_contour_map(depth_map, rois, save_path, x_scale, y_scale, origin_x, origin_y, pad_px=5):
    """
    Contour plot over the union of *rois* only.  Pixels outside the ROIs are
    masked (transparent).  Axes are labelled in metres.
    """
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    xs, ys, xe, ye = [], [], [], []
    for x, y, w, h in rois:
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        mask[y0:y1, x0:x1] = True
        xs.append(x0); ys.append(y0); xe.append(x1); ye.append(y1)

    x_min = max(0, min(xs) - pad_px)
    y_min = max(0, min(ys) - pad_px)
    x_max = min(W, max(xe) + pad_px)
    y_max = min(H, max(ye) + pad_px)

    Z = depth_map[y_min:y_max, x_min:x_max].copy().astype(float)
    M = mask[y_min:y_max, x_min:x_max]
    Z[~M] = np.nan

    x_pix = np.arange(x_min, x_max)
    y_pix = np.arange(y_min, y_max)
    Xm = (x_pix - origin_x) * x_scale
    Ym = (y_pix - origin_y) * (-y_scale)
    X, Y = np.meshgrid(Xm, Ym)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0.0)

    finite_vals = Z[np.isfinite(Z)]
    vmin = float(np.nanmin(Z)) if finite_vals.size else 0.0
    vmax = float(np.nanmax(Z)) if finite_vals.size else 1.0
    levels = np.linspace(vmin, vmax, 12)

    filled = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="both")
    lines = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.6)
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.1f", inline_spacing=8)

    cbar = plt.colorbar(filled, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Depth (m)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Average Physical Depth Contour Map (ROIs)", fontsize=13, pad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved ROI-only contour map to: {save_path}")


def save_physical_depth_to_csv(depth_map, x_scale, y_scale, origin_x, origin_y, file_path):
    """
    Save the physically calibrated depth map to a CSV with columns
    ``X_m``, ``Y_m``, ``Depth_m`` — one row per pixel.
    """
    H, W = depth_map.shape
    X = (np.arange(W) - origin_x) * x_scale
    Y = (np.arange(H) - origin_y) * (-y_scale)
    X_grid, Y_grid = np.meshgrid(X, Y)

    pd.DataFrame({
        "X_m": X_grid.flatten(),
        "Y_m": Y_grid.flatten(),
        "Depth_m": depth_map.flatten(),
    }).to_csv(file_path, index=False)
    print(f"Saved physical depth map data to '{file_path}'")


# ===========================================================================
# Interactive helpers
# ===========================================================================

def select_two_rois(video_path):
    """
    Display the first frame of *video_path* and ask the user to select two
    rectangular ROIs via ``cv2.selectROI``.  Returns ``[roi1, roi2]`` where
    each ROI is ``(x, y, w, h)`` in frame coordinates after 90-degree CCW
    rotation.

    Raises ``IOError`` if the video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise IOError(f"Cannot read first frame from: {video_path}")

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print("Select first ROI, then press ENTER or SPACE. Press C to cancel.")
    roi1 = cv2.selectROI("Select ROI 1", frame, showCrosshair=True, fromCenter=False)
    print("Select second ROI, then press ENTER or SPACE. Press C to cancel.")
    roi2 = cv2.selectROI("Select ROI 2", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    return [roi1, roi2]


def tune_gamma(frame, roi):
    """
    Interactive trackbar widget for tuning gamma on a single ROI crop.

    The trackbar maps positions 1--100 to gamma values 0.1--10.0.
    Press ENTER or SPACE to accept; ESC to cancel (returns 1.0).
    """
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return 1.0

    roi_crop = frame[y : y + h, x : x + w].copy()
    if roi_crop.size == 0:
        return 1.0

    cv2.namedWindow("Gamma Adjustment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gamma Adjustment", 800, 500)
    cv2.createTrackbar("Gamma x10", "Gamma Adjustment", 10, 100, lambda v: None)

    print("\nAdjust the 'Gamma x10' slider (0.1-10.0). Press SPACE or ENTER when done.\n")

    while True:
        pos = cv2.getTrackbarPos("Gamma x10", "Gamma Adjustment")
        gamma_val = max(0.1, pos / 10.0)
        gamma_corrected = apply_gamma_correction(roi_crop, gamma=gamma_val)

        left = roi_crop.copy()
        right = gamma_corrected.copy()
        cv2.putText(left, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(right, f"Gamma = {gamma_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        display = np.hstack([left, right])
        cv2.imshow("Gamma Adjustment", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32):
            break
        elif key == 27:
            cv2.destroyWindow("Gamma Adjustment")
            return 1.0

    cv2.destroyWindow("Gamma Adjustment")
    print(f"Selected gamma value: {gamma_val:.1f}")
    return gamma_val


def get_calibration_points(image):
    """
    Collect four mouse-click points from *image* for spatial calibration:

      1. X origin  (left reference point)
      2. X reference (right reference point, *DAM_WIDTH_METRES* away)
      3. Y origin  (top reference point)
      4. Y reference (bottom reference point, *DAM_HEIGHT_METRES* below)

    Waits until four points have been clicked, then closes the window.
    Returns a list of four ``(x, y)`` tuples.
    """
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            print(f"Point {len(points) + 1}: ({x}, {y})")
            points.append((x, y))
            if len(points) == 4:
                print("All 4 points collected.")

    cv2.namedWindow("Click calibration points", cv2.WINDOW_NORMAL)
    cv2.imshow("Click calibration points", image.copy())
    cv2.setMouseCallback("Click calibration points", click_event)
    print("Click 4 points in order: X-origin, X-ref, Y-origin, Y-ref.")

    while len(points) < 4:
        if cv2.waitKey(30) & 0xFF == 27:  # ESC exits early
            break

    cv2.destroyAllWindows()
    return points


# ===========================================================================
# Main processing pipeline
# ===========================================================================

def process_video(
    input_path,
    out_sfs_heatmap=OUTPUT_SFS_HEATMAP,
    out_physical_heatmap=OUTPUT_PHYSICAL_HEATMAP,
    out_contour_map=OUT_CONTOUR_MAP,
    out_3d_png=OUT_3D_DEPTH_PNG,
    out_profile_png=OUT_DEPTH_PROFILE,
    max_frames=None,
):
    """
    Full SfS pipeline: ROI selection, gamma tuning, per-frame depth estimation,
    video writing, calibration, and diagnostic plot generation.

    Parameters
    ----------
    input_path : str
        Path to the input video file.
    out_* : str
        Output file paths for each artefact.
    max_frames : int or None
        Maximum number of frames to process.  ``None`` processes all frames.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Video not found: {input_path}")

    rois = select_two_rois(input_path)
    print(f"Selected ROIs: {rois}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_max = total_frames if max_frames is None else min(max_frames, total_frames)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, first_frame_raw = cap.read()
    if not ret or first_frame_raw is None:
        cap.release()
        raise IOError("Cannot read first frame")

    first_frame = cv2.rotate(first_frame_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h_frame, w_frame = first_frame.shape[:2]

    print("Tuning gamma for first (upper) ROI")
    gamma_top = tune_gamma(first_frame, rois[0])
    print("Tuning gamma for second (lower) ROI")
    gamma_bottom = tune_gamma(first_frame, rois[1])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    colorbar_sfs = create_colorbar(h_frame, cv2.COLORMAP_VIRIDIS, 0, 1, label="Relative Depth")
    colorbar_physical = create_colorbar(h_frame, cv2.COLORMAP_INFERNO, 0, 18.0, label="Depth (m)")

    frame_width_sfs = w_frame + colorbar_sfs.shape[1]
    frame_width_phys = w_frame + colorbar_physical.shape[1]

    writer_fps = fps / max(1, frame_stride)
    vw_sfs = cv2.VideoWriter(out_sfs_heatmap, fourcc, writer_fps, (frame_width_sfs, h_frame))
    vw_physical = cv2.VideoWriter(out_physical_heatmap, fourcc, writer_fps, (frame_width_phys, h_frame))

    if not vw_sfs.isOpened() or not vw_physical.isOpened():
        cap.release()
        raise IOError("Failed to open one or more VideoWriter outputs")

    # Rewind to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_z = None
    frame_idx = 0
    saved_frames = 0
    all_h_water = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if saved_frames >= effective_max:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if (frame_idx % frame_stride) != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

        for i, (x, y, w_roi, h_roi) in enumerate(rois):
            gamma_val = gamma_top if i == 0 else gamma_bottom
            roi_section = gray[y : y + h_roi, x : x + w_roi]
            if roi_section.size > 0:
                gray[y : y + h_roi, x : x + w_roi] = apply_gamma_correction(
                    roi_section, gamma=gamma_val
                )

        I = gray.astype(np.float32) / 255.0
        mask = (gray < SPECULAR_THRESHOLD).astype(np.float32)

        roi_mask = np.zeros_like(mask)
        for x, y, w_roi, h_roi in rois:
            roi_mask[int(y) : int(y + h_roi), int(x) : int(x + w_roi)] = 1.0
        mask *= roi_mask

        z_init = prev_z if (use_temporal_init and prev_z is not None) else None
        z = run_sfs_on_frame(
            I, z_init=z_init, s=s, rho=rho,
            lam=SMOOTHNESS_LAMBDA, alpha=LEARNING_RATE_ALPHA,
            iters=SFS_ITERATIONS, mask=mask,
        )
        prev_z = z.copy()

        h_water, _ = convert_sfs_to_physical_depth(z)
        all_h_water.append(h_water)

        z_normed = ((z - z.min()) / (z.max() - z.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_sfs = cv2.applyColorMap(z_normed, cv2.COLORMAP_VIRIDIS)
        # Ensure colorbar height matches frame height before hstack
        if colorbar_sfs.shape[0] != heatmap_sfs.shape[0]:
            colorbar_sfs = create_colorbar(heatmap_sfs.shape[0], cv2.COLORMAP_VIRIDIS, 0, 1, label="Relative Depth")
        heatmap_sfs = np.hstack([heatmap_sfs, colorbar_sfs])

        h_normed = ((h_water - h_water.min()) / (h_water.max() - h_water.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_physical = cv2.applyColorMap(h_normed, cv2.COLORMAP_INFERNO)
        if colorbar_physical.shape[0] != heatmap_physical.shape[0]:
            colorbar_physical = create_colorbar(heatmap_physical.shape[0], cv2.COLORMAP_INFERNO, 0, 18.0, label="Depth (m)")
        heatmap_physical = np.hstack([heatmap_physical, colorbar_physical])

        cv2.imshow("SfS Depth Heatmap", heatmap_sfs)
        cv2.imshow("Physical Depth Heatmap", heatmap_physical)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        vw_sfs.write(heatmap_sfs)
        vw_physical.write(heatmap_physical)

        saved_frames += 1
        frame_idx += 1
        if saved_frames % 10 == 0:
            print(f"Processed {saved_frames}/{effective_max} frames...")

    cap.release()
    vw_sfs.release()
    vw_physical.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Processed {saved_frames} frames.")

    # ------------------------------------------------------------------
    # Spatial calibration — uses the first frame captured at the start
    # ------------------------------------------------------------------
    print("\nOpening calibration window. Click 4 points: X-origin, X-ref, Y-origin, Y-ref.")
    points = get_calibration_points(first_frame)
    if len(points) < 4:
        print(
            f"Warning: calibration requires 4 points but only {len(points)} were clicked. "
            "Skipping spatial plots.",
            file=sys.stderr,
        )
        _write_summary(saved_frames, out_sfs_heatmap, out_physical_heatmap, out_3d_png, out_contour_map, out_profile_png)
        return

    p1x, p2x, p1y, p2y = points
    dx_pixels = max(1, abs(p2x[0] - p1x[0]))
    dy_pixels = max(1, abs(p2y[1] - p1y[1]))

    x_scale = DAM_WIDTH_METRES / dx_pixels    # metres per pixel (horizontal)
    y_scale = DAM_HEIGHT_METRES / dy_pixels   # metres per pixel (vertical)
    origin_x, origin_y = p1x
    print(
        f"Calibration complete:\n"
        f"  x_scale = {x_scale:.5f} m/px\n"
        f"  y_scale = {y_scale:.5f} m/px"
    )

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    if all_h_water:
        avg_h_water = np.mean(all_h_water, axis=0)

        print("Plotting average physical depth map (3D)...")
        plot_average_physical_depth(avg_h_water, out_3d_png, x_scale, y_scale, origin_x, origin_y)

        print("Plotting contour map...")
        plot_contour_map(avg_h_water, rois, out_contour_map, x_scale, y_scale, origin_x, origin_y)

        print("Saving physical depth map to CSV...")
        save_physical_depth_to_csv(
            avg_h_water, x_scale, y_scale, origin_x, origin_y, "physical_depth_map.csv"
        )

    if prev_z is not None and all_h_water:
        last_z = prev_z
        last_h_water = all_h_water[-1]
        row_idx = last_z.shape[0] // 2

        x_px = np.arange(last_z.shape[1])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_px, last_z[row_idx, :], label="SfS Depth (relative units)", color="blue")
        ax.plot(x_px, last_h_water[row_idx, :], label="Physical Depth (m)", color="red")
        ax.set_xlabel("Pixel along row")
        ax.set_ylabel("Depth")
        ax.set_title(f"Depth profile along row {row_idx} (last processed frame)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_profile_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved depth profile to: {out_profile_png}")

    _write_summary(saved_frames, out_sfs_heatmap, out_physical_heatmap, out_3d_png, out_contour_map, out_profile_png)


def _write_summary(saved_frames, *paths):
    labels = [
        "SfS Heatmap", "Physical Depth Heatmap",
        "3D Depth Map", "Contour Map", "Depth Profile",
    ]
    print(f"\nDone. Processed {saved_frames} frames.")
    print("Output files:")
    for label, path in zip(labels, paths):
        print(f"  {label}: {path}")


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Shape-from-Shading depth estimation from fluid-dynamics video."
    )
    parser.add_argument(
        "--video", default="IMGP9471.MOV",
        help="Path to the input video file (default: IMGP9471.MOV)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process (default: all frames)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        print("Usage: python SFSImplementation.py --video /path/to/video.MOV", file=sys.stderr)
        sys.exit(1)

    process_video(
        input_path=args.video,
        out_sfs_heatmap=OUTPUT_SFS_HEATMAP,
        out_physical_heatmap=OUTPUT_PHYSICAL_HEATMAP,
        out_contour_map=OUT_CONTOUR_MAP,
        out_3d_png=OUT_3D_DEPTH_PNG,
        out_profile_png=OUT_DEPTH_PROFILE,
        max_frames=args.max_frames,
    )
