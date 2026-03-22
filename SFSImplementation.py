import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from preprocessing import apply_gamma_correction
import pandas as pd

# Video input
INPUT_VIDEO = "IMGP9471.MOV"

# Output paths
OUTPUT_SFS_HEATMAP = "sfs_depth_heatmap.mp4"
OUTPUT_PHYSICAL_HEATMAP = "physical_depth_heatmap.mp4"
OUT_CONTOUR_MAP = "physical_depth_contour_map.png"
OUT_3D_DEPTH_PNG = "average_physical_depth_map.png"
OUT_DEPTH_PROFILE = "depth_profile_physical_vs_sfs.png"

# Photometric model
rho = 1.0 
LIGHT_DIRECTION = np.array([0.2, 0.1, 0.97]) # Light source direction
s = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)

# Optimization parameters
SMOOTHNESS_LAMBDA = 1.0
LEARNING_RATE_ALPHA = 0.05
SFS_ITERATIONS = 400
GAUSSIAN_KERNEL = (7, 7)
SPECULAR_THRESHOLD = 245

# Dam Reference Points
Z_TOP_M = 93.0
Z_TURB_M = 84.0
Z_END_M = 75.0

# Temporal processing
use_temporal_init = True
frame_stride = 1
cap = cv2.VideoCapture(INPUT_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
MAX_FRAMES = total_frames # Set desired number of frames

def forward_gradients(z):
    zx = np.zeros_like(z)
    zy = np.zeros_like(z)
    zx[:, :-1] = z[:, 1:] - z[:, :-1]
    zy[:-1, :] = z[1:, :] - z[:-1, :]
    return zx, zy

def laplacian(z):
    lap = -4 * z.copy()
    lap += np.pad(z[1:, :], ((0,1),(0,0)), mode='constant')
    lap += np.pad(z[:-1, :], ((1,0),(0,0)), mode='constant')
    lap += np.pad(z[:, 1:], ((0,0),(0,1)), mode='constant')
    lap += np.pad(z[:, :-1], ((0,0),(1,0)), mode='constant')
    return lap

"""
Performs Shape-from-Shading on a single frame.
Iteratively refines the depth map z to minimise the difference between
the observed intensities I and the intensities predicted by the SfS model.
"""
def run_sfs_on_frame(I, z_init=None, s=s, rho=rho, lam=SMOOTHNESS_LAMBDA, alpha=LEARNING_RATE_ALPHA, iters=200, mask=None):
    H, W = I.shape
    if mask is None:
        mask = np.ones((H, W), dtype=np.float32)

    z = np.zeros((H, W), dtype=np.float32) if z_init is None else z_init.astype(np.float32).copy()
    z_outside_fixed_value = 0.0
    z[mask == 0] = z_outside_fixed_value

    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    eps = 1e-8

    for it in range(iters):
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
        z[mask == 0] = z_outside_fixed_value

    return z

"""
Converts the relative SfS depth map to physical water depth in metres.
"""
def convert_sfs_to_physical_depth(z_sfs, z_top_m = Z_TOP_M, z_turb_m = Z_TURB_M, z_end_m = Z_END_M):
    z_norm = (z_sfs - z_sfs.min()) / (z_sfs.max() - z_sfs.min() + 1e-8)
    z_physical = z_norm * (z_top_m  - z_turb_m) + z_turb_m
    h_water = z_physical - z_end_m
    h_water[h_water < 0] = 0
    mask = z_physical >= z_end_m
    return h_water, mask

def create_colorbar(height, colormap, min_val, max_val, width=50, label="Depth (m)"):
    """
    Create a vertical colorbar with units and correct array dimensions.
    """
    # Gradient image
    gradient = np.linspace(0, 255, height).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(gradient, colormap)
    colorbar = cv2.resize(colorbar, (width, height))

    # Tick marks
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)

    tick_positions = np.linspace(0, height - 1, 5).astype(int)
    tick_values = np.linspace(max_val, min_val, 5)  # top to bottom

    for pos, val in zip(tick_positions, tick_values):
        y = int(pos)
        cv2.putText(
            colorbar,
            f"{val:.2f} m",
            (5, y + 5),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    # Create label bar of same height
    label_width = 60
    label_img = np.zeros((height, label_width, 3), dtype=np.uint8)

    # Write vertical label text centered
    text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
    # Write text horizontally on temporary image
    temp_img = np.zeros((label_width, height, 3), dtype=np.uint8)
    text_x = (height - text_size[0]) // 2
    text_y = (label_width + text_size[1]) // 2
    cv2.putText(temp_img, label, (text_x, text_y), font, 0.6, text_color, 1, cv2.LINE_AA)
    # Rotate it to make it vertical
    label_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    label_img = cv2.resize(label_img, (label_width, height))

    # Combine label + colorbar
    full_bar = np.hstack([label_img, colorbar])
    return full_bar

def select_two_rois(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Cannot read first frame for ROI selection")

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print("Select first ROI, then press ENTER or SPACE. Press C to cancel.")
    roi1 = cv2.selectROI("Select ROI 1", frame, showCrosshair=True, fromCenter=False)
    print("Select second ROI, then press ENTER or SPACE. Press C to cancel.")
    roi2 = cv2.selectROI("Select ROI 2", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    return [roi1, roi2]


def tune_gamma(frame, roi):
    """
    Opens a window with a numeric gamma slider for fine-tuning contrast on the ROI.
    """
    x, y, w, h = map(int, roi)
    roi_crop = frame[y:y+h, x:x+w].copy()

    cv2.namedWindow("Gamma Adjustment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gamma Adjustment", 800, 500)
    cv2.createTrackbar("Gamma", "Gamma Adjustment", 10, 100, lambda x: None)  

    print("\nAdjust the 'Gamma' slider (1.0–10.0). Press SPACE or ENTER when done.\n")

    while True:
        gamma_val = cv2.getTrackbarPos("Gamma", "Gamma Adjustment") / 10.0
        gamma_corrected = apply_gamma_correction(roi_crop, gamma=gamma_val)
        
        display = np.hstack([
            cv2.putText(roi_crop.copy(), f"Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2),
            cv2.putText(gamma_corrected, f"Gamma = {gamma_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        ])

        cv2.imshow("Gamma Adjustment", display)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]:  # ENTER or SPACE
            break
        elif key == 27:  # ESC cancels
            cv2.destroyWindow("Gamma Adjustment")
            return 1.0  # default gamma

    cv2.destroyWindow("Gamma Adjustment")
    print(f"Selected gamma value: {gamma_val:.1f}")
    return gamma_val

def get_calibration_points(image):
    """
    Click four points to define spatial scaling:
    1. X origin (0, 0)
    2. X reference (known width apart, e.g. 12.25 m) (Found from Video)
    3. Y origin (same as #1 ideally)
    4. Y reference (known vertical distance, e.g. 21 m below) (Found from Video)
    """
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Point {len(points)+1}: ({x}, {y})")
            points.append((x, y))

    clone = image.copy()
    cv2.imshow("Click calibration points", clone)
    cv2.setMouseCallback("Click calibration points", click_event, points)
    print("Click 4 points in order: X-origin, X-ref, Y-origin, Y-ref.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points

def plot_average_physical_depth(depth_map, save_path, x_scale, y_scale, origin_x, origin_y):
    H, W = depth_map.shape

    # Convert pixel indices to metres
    X = (np.arange(W) - origin_x) * x_scale
    Y = (np.arange(H) - origin_y) * (-y_scale)
    X, Y = np.meshgrid(X, Y)

    Z = depth_map

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='inferno', linewidth=0, antialiased=True)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_title('Average Physical Depth Map (Metre Calibrated)')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Depth (m)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved 3D depth map (calibrated) to: {save_path}")


def plot_contour_map(depth_map, rois, save_path, x_scale, y_scale, origin_x, origin_y, pad_px=5):
    """
    Contour plot over the union of the selected ROIs only.
    Outside-ROI pixels are transparent (no green background).
    Axes are in metres using your calibration.
    """
    H, W = depth_map.shape

    # Build a mask for the two ROIs (union)
    mask = np.zeros((H, W), dtype=bool)
    xs, ys, xe, ye = [], [], [], []
    for (x, y, w, h) in rois:
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + w), int(y + h)
        mask[y0:y1, x0:x1] = True
        xs.append(x0); ys.append(y0); xe.append(x1); ye.append(y1)

    # Crop ROI
    x_min = max(0, min(xs) - pad_px)
    y_min = max(0, min(ys) - pad_px)
    x_max = min(W, max(xe) + pad_px)
    y_max = min(H, max(ye) + pad_px)

    Z = depth_map[y_min:y_max, x_min:x_max].copy()
    M = mask[y_min:y_max, x_min:x_max]

    # Mask everything outside the ROIs to NaN so it's transparent
    Z[~M] = np.nan

    x_pix = np.arange(x_min, x_max)
    y_pix = np.arange(y_min, y_max)
    Xm = (x_pix - origin_x) * x_scale
    Ym = (y_pix - origin_y) * (-y_scale)
    X, Y = np.meshgrid(Xm, Ym)

    plt.figure(figsize=(10, 7), facecolor='white')
    ax = plt.gca()

    # Transparent for NaN cells
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0.0)

    # Choose reasonable level count
    finite_vals = Z[np.isfinite(Z)]
    vmin = np.nanmin(Z) if finite_vals.size else 0.0
    vmax = np.nanmax(Z) if finite_vals.size else 1.0
    levels = np.linspace(vmin, vmax, 12)

    filled = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
    lines  = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.6)
    ax.clabel(lines, inline=True, fontsize=8, fmt='%.1f', inline_spacing=8)

    cbar = plt.colorbar(filled, pad=0.02, fraction=0.03)
    cbar.set_label('Depth (m)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Average Physical Depth Contour Map (ROIs)', fontsize=13, pad=10)
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_facecolor('white')

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved ROI-only contour map to: {save_path}")

def plot_reconstructed_surface(z_surface, save_path):
    """
    Visualise the reconstructed surface (SfS output) for the last frame.
    """
    H, W = z_surface.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, z_surface, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Depth (relative units)")
    ax.set_title("Reconstructed surface (last frame)")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Relative Depth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved reconstructed surface (last frame) to: {save_path}")

def save_physical_depth_to_csv(depth_map, x_scale, y_scale, origin_x, origin_y, file_path):
    """
    Saves the physical depth map (metre-calibrated) to a CSV with X, Y, and Depth (m) columns.
    Each row corresponds to one pixel coordinate in real-world metres.
    """
    H, W = depth_map.shape
    X = (np.arange(W) - origin_x) * x_scale
    Y = (np.arange(H) - origin_y) * (-y_scale)
    X_grid, Y_grid = np.meshgrid(X, Y)

    df = pd.DataFrame({
        "X_m": X_grid.flatten(),
        "Y_m": Y_grid.flatten(),
        "Depth_m": depth_map.flatten()
    })

    df.to_csv(file_path, index=False)
    print(f"Saved physical depth map data to '{file_path}'")

def process_video(input_path, out_sfs_heatmap, out_physical_heatmap, out_contour_map, out_3d_png, out_profile_png):
    rois = select_two_rois(input_path)
    print(f"Selected ROIs: {rois}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame")

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = frame.shape[:2]

    print("Tuning gamma for first ROI")
    gamma_top = tune_gamma(frame, rois[0])

    print("\nTune gamma for the lower ROI")
    gamma_bottom = tune_gamma(frame, rois[1])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # safer codec on macOS

    # Create colorbars first
    colorbar_sfs = create_colorbar(h, cv2.COLORMAP_VIRIDIS, min_val=0, max_val=1, label="Relative Depth")
    colorbar_physical = create_colorbar(h, cv2.COLORMAP_INFERNO, min_val=0, max_val=18.0, label="Depth (m)")
    # Dynamically match writer width to actual frame + colorbar size
    frame_width_sfs = w + colorbar_sfs.shape[1]
    frame_width_phys = w + colorbar_physical.shape[1]
    vw_sfs = cv2.VideoWriter(out_sfs_heatmap, fourcc, fps / max(1, frame_stride), (frame_width_sfs, h))
    vw_physical = cv2.VideoWriter(out_physical_heatmap, fourcc, fps / max(1, frame_stride), (frame_width_phys, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_z = None
    frame_idx = 0
    saved_frames = 0
    all_h_water = []

    while True:
        ret, frame = cap.read()
        if not ret or saved_frames >= MAX_FRAMES:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if (frame_idx % frame_stride) != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

        # Apply gamma correction
        for i, (x, y, w_roi, h_roi) in enumerate(rois):
            gamma_val = gamma_top if i == 0 else gamma_bottom
            roi_section = gray[y:y+h_roi, x:x+w_roi]
            gray[y:y+h_roi, x:x+w_roi] = apply_gamma_correction(roi_section, gamma=gamma_val)

        I = gray.astype(np.float32) / 255.0
        mask = (gray < SPECULAR_THRESHOLD).astype(np.float32)

        roi_mask = np.zeros_like(mask)
        for (x, y, w_roi, h_roi) in rois:
            roi_mask[int(y):int(y + h_roi), int(x):int(x + w_roi)] = 1.0
        mask *= roi_mask

        z_init = prev_z if (use_temporal_init and prev_z is not None) else None
        z = run_sfs_on_frame(I, z_init=z_init, s=s, rho=rho, lam=SMOOTHNESS_LAMBDA, alpha=LEARNING_RATE_ALPHA, iters=SFS_ITERATIONS, mask=mask)
        prev_z = z.copy()

        h_water, _ = convert_sfs_to_physical_depth(z)
        all_h_water.append(h_water)

        z_normed = ((z - z.min()) / (z.max() - z.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_sfs = cv2.applyColorMap(z_normed, cv2.COLORMAP_VIRIDIS)
        heatmap_sfs = np.hstack([heatmap_sfs, colorbar_sfs])

        h_normed = ((h_water - h_water.min()) / (h_water.max() - h_water.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_physical = cv2.applyColorMap(h_normed, cv2.COLORMAP_INFERNO)
        heatmap_physical = np.hstack([heatmap_physical, colorbar_physical])

        cv2.imshow("SfS Depth Heatmap", heatmap_sfs)
        cv2.imshow("Physical Depth Heatmap", heatmap_physical)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        vw_sfs.write(heatmap_sfs)
        vw_physical.write(heatmap_physical)

        saved_frames += 1
        frame_idx += 1
        if saved_frames % 2 == 0:
            print(f"Processed {saved_frames}/{MAX_FRAMES} frames...")

    # Calibration steps
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame for calibration.")

    first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    points = get_calibration_points(first_frame)
    if len(points) != 4:
        raise ValueError("Need exactly 4 points: X-origin, X-ref, Y-origin, Y-ref")

    p1x, p2x, p1y, p2y = points

    # Pixel distances
    dx_pixels = abs(p2x[0] - p1x[0])   # only X difference
    dy_pixels = abs(p2y[1] - p1y[1])   # only Y difference (avoids human error in selecting the origin points)

    # Known real-world distances (metres)
    DAM_WIDTH_METRES = 12.25
    DAM_HEIGHT_METRES = 21.0 # Down to point of interest

    # Scaling factors (metres per pixel)
    x_scale = DAM_WIDTH_METRES / dx_pixels
    y_scale = DAM_HEIGHT_METRES / dy_pixels

    # Set origin
    origin_x, origin_y = p1x
    print(f"Calibration complete:\n  x_scale = {x_scale:.5f} m/px\n  y_scale = {y_scale:.5f} m/px")

    if len(all_h_water) > 0:
        avg_h_water = np.mean(all_h_water, axis=0)
        print("Plotting average physical depth map (3D)...")
        plot_average_physical_depth(avg_h_water, out_3d_png, x_scale, y_scale, origin_x, origin_y)
        
        print("Plotting contour map...")
        plot_contour_map(avg_h_water, rois, out_contour_map, x_scale, y_scale, origin_x, origin_y)

        print("Saving physical depth map to CSV...")
        save_physical_depth_to_csv(avg_h_water, x_scale, y_scale, origin_x, origin_y, "physical_depth_map.csv")

    # Depth profile plot
    if prev_z is not None and len(all_h_water) > 0:
        last_z = prev_z
        last_h_water = all_h_water[-1]

        row_idx = last_z.shape[0] // 2  # Use the central row
        x = np.arange(last_z.shape[1])
        sfs_depth_row = last_z[row_idx, :]
        physical_depth_row = last_h_water[row_idx, :]

        plt.figure(figsize=(10, 4))
        plt.plot(x, sfs_depth_row, label='SfS Depth (relative units)', color='blue')
        plt.plot(x, physical_depth_row, label='Physical Depth (m)', color='red')
        plt.xlabel("Pixel along row")
        plt.ylabel("Depth")
        plt.title(f"Depth profile along row {row_idx} (last frame)")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_profile_png)
        plt.show()
        print(f"Saved depth profile to: {out_profile_png}")

    cap.release()
    vw_sfs.release()
    vw_physical.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {saved_frames} frames.")
    print(f"Output files:")
    print(f"  - SfS Heatmap: {out_sfs_heatmap}")
    print(f"  - Physical Depth Heatmap: {out_physical_heatmap}")
    print(f"  - 3D Depth Map: {out_3d_png}")
    print(f"  - Contour Map: {out_contour_map}")
    print(f"  - Depth Profile: {out_profile_png}")

if __name__ == "__main__":
    process_video(INPUT_VIDEO, OUTPUT_SFS_HEATMAP, OUTPUT_PHYSICAL_HEATMAP, OUT_CONTOUR_MAP, OUT_3D_DEPTH_PNG, OUT_DEPTH_PROFILE)