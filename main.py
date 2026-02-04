import sys, csv, os
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl

from preprocessing import (
    rotate_image, resize_to_screen, apply_clahe, apply_gamma_correction,
    point_to_point, select_gamma_for_roi
)

from SFSImplementation import forward_gradients, run_sfs_on_frame

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QComboBox, QCheckBox, QPushButton, QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


video_path = "IMGP9471.MOV"
validation_path = "ideal.csv"
output_path = "velocity_output.csv"

num_regions = 2
sfs_influences = [0.8, 0.7]
penalty_factors = [1, 1]
rotation_angle = 270
known_width_m = 12.25

height_breaks = [544, 555, 579, 613, 648, 691, 731, 768, 813]
height_scales = [25.84, 31.67, 35.84, 39.09, 42.89, 45.86, 48.28, 50.30, 51.34]

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

fps = video.get(cv2.CAP_PROP_FPS) or 25.0
ok, first_frame_raw = video.read()
if not ok or first_frame_raw is None:
    raise IOError("Failed to read first frame")

rotated_for_cal = rotate_image(first_frame_raw.copy(), rotation_angle)
mpp = point_to_point(rotated_for_cal, known_metres=known_width_m)
if mpp is None:
    print("Calibration cancelled/failed. Falling back to estimate 0.039 m/px.")
    mpp = 0.039
else:
    print(f"Calibration result: {mpp:.8f} m/px")

rois, gammas = [], []
rotated_for_roi = rotate_image(first_frame_raw.copy(), rotation_angle)
resized, scale = resize_to_screen(rotated_for_roi)

for i in range(num_regions):
    roi = cv2.selectROI(f"Select ROI {i+1}", resized, showCrosshair=True)
    x, y, w, h = roi
    rois.append((int(x/scale), int(y/scale), int(w/scale), int(h/scale)))
    gsel = select_gamma_for_roi(resized, (x, y, w, h), initial_gamma=1.0, window_name=f"Gamma ROI {i+1}")
    gammas.append(1.0 if gsel is None else float(gsel))

cv2.destroyAllWindows()

video.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, first_frame = video.read()
if not ok or first_frame is None:
    raise IOError("Failed to re-read first frame")
first_frame = rotate_image(first_frame, rotation_angle)

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_gray = apply_clahe(prev_gray)
prev_gray = apply_gamma_correction(prev_gray, gammas[0])

cv2.namedWindow("Optical Flow Quiver Plot")

velocity_sums = [None for _ in range(num_regions)]
frame_counts = [0 for _ in range(num_regions)]
vel_history = [[] for _ in range(num_regions)]

while True:
    working, frame = video.read()
    if not working or frame is None:
        break

    frame = rotate_image(frame, rotation_angle)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)

    for i, (x, y, w, h) in enumerate(rois):
        roi_prev = prev_gray[y:y+h, x:x+w]
        roi_curr = gray[y:y+h, x:x+w]
        roi_curr = apply_gamma_correction(roi_curr, gammas[i])

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        sfs_float = roi_curr.astype(np.float32) / 255.0
        z = run_sfs_on_frame(sfs_float, iters=30)
        zx, zy = forward_gradients(z)
        downslope = np.stack([-zx, -zy], axis=-1)
        downslope /= (np.linalg.norm(downslope, axis=-1, keepdims=True) + 1e-8)

        flow_unit = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-8)
        alignment = np.sum(flow_unit * downslope, axis=-1)

        penalty = penalty_factors[i]
        raw_weights = np.clip((alignment + 1) / 2, 0, 1)
        raw_weights[alignment < 0] *= (1 - penalty)
        influence = sfs_influences[i]
        weights = 1 - influence * (1 - raw_weights)
        flow_refined = flow * weights[..., None]

        fx, fy = flow_refined[..., 0], flow_refined[..., 1]
        mag, _ = cv2.cartToPolar(fx, fy, angleInDegrees=True)
        velocity = mag * fps * mpp

        yy_coords = np.arange(y, y + h)
        row_angles_deg = np.interp(yy_coords, height_breaks, height_scales)
        row_scales = np.sin(np.radians(row_angles_deg))
        velocity /= row_scales[:, None]

        if velocity_sums[i] is None:
            velocity_sums[i] = np.zeros_like(velocity, dtype=np.float32)
        velocity_sums[i] += velocity.astype(np.float32)
        frame_counts[i] += 1
        vel_history[i].append(velocity.astype(np.float32))

        avg_velocity = float(np.mean(velocity))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"ROI {i+1}: {avg_velocity:.2f} m/s (gamma={gammas[i]:.2f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        step = 12
        for yy in range(0, h, step):
            for xx in range(0, w, step):
                dx = int(flow_refined[yy, xx, 0])
                dy = int(flow_refined[yy, xx, 1])
                if dx*dx + dy*dy > 1:
                    color = (0, 0, 255) if alignment[yy, xx] >= 0 else (0, 165, 255)
                    cv2.arrowedLine(frame, (x + xx, y + yy), (x + xx + dx, y + yy + dy),
                                    color, 1, tipLength=0.4)

    display, _ = resize_to_screen(frame.copy())
    cv2.imshow("Optical Flow Quiver Plot", display)
    prev_gray = gray.copy()

    key = cv2.waitKeyEx(10)
    if key in (27, ord('q'), ord('Q')):
        break
    if cv2.getWindowProperty("Optical Flow Quiver Plot", cv2.WND_PROP_VISIBLE) < 1:
        break

def get_avg_velocity(velocity_sums, frame_counts, rois, mpp, increment_m=1.0):
    height_data = {}
    current_height_m = 0

    for i, (x, y, w, h) in enumerate(rois):
        avg_velocity_map = velocity_sums[i] / max(1, frame_counts[i])
        roi_height_m = h * mpp
        num_increments = int(np.floor(roi_height_m / increment_m))

        for inc in range(num_increments):
            start_y_m = current_height_m + inc * increment_m
            end_y_m = start_y_m + increment_m
            start_y_px = int((start_y_m - current_height_m) / mpp)
            end_y_px = int((end_y_m - current_height_m) / mpp)

            if start_y_px >= h:
                break
            end_y_px = min(end_y_px, h)

            region_velocity = avg_velocity_map[start_y_px:end_y_px, :]
            avg_velocity = np.mean(region_velocity)
            height_key = round(start_y_m, 1)
            if height_key not in height_data:
                height_data[height_key] = {"Velocity m/s": avg_velocity}
            else:
                height_data[height_key]["Velocity m/s"] = max(
                    height_data[height_key]["Velocity m/s"], avg_velocity
                )
        current_height_m += roi_height_m
    return height_data

velocity_data = get_avg_velocity(velocity_sums, frame_counts, rois, mpp, increment_m=1.0)

with open(output_path, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Depth m", "Velocity m/s"])
    writer.writeheader()
    for height, data in velocity_data.items():
        writer.writerow({"Depth m": height, "Velocity m/s": data["Velocity m/s"]})

print(f"Velocity data exported to {output_path}")

video.release()
cv2.destroyAllWindows()

h_full, w_full = first_frame.shape[:2]
avg_maps = []
all_vals = []
for i, (x, y, w, h) in enumerate(rois):
    if frame_counts[i] == 0:
        amap = np.zeros((h, w), dtype=np.float32)
    else:
        amap = velocity_sums[i] / float(frame_counts[i])
    avg_maps.append(amap)
    if amap.size:
        all_vals.append(amap.reshape(-1))

global_cap = float(np.percentile(np.concatenate(all_vals), 99)) if len(all_vals) else 1.0

def build_overlay_from_avg_maps(colormap_name="JET", cap_val=None, add_scale_bar=False, scale_m=1.0):
    cmap_table = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }
    cm_choice = cmap_table.get(colormap_name.upper(), cv2.COLORMAP_JET)
    combined_heatmap = np.zeros((h_full, w_full, 3), dtype=np.uint8)
    combined_mask = np.zeros((h_full, w_full), dtype=bool)
    for i, (x, y, w, h) in enumerate(rois):
        amap = avg_maps[i]
        if amap.size:
            cap = (cap_val if (cap_val is not None and cap_val > 0) else max(1e-6, float(np.max(amap))))
            capped = np.clip(amap, 0, cap)
            heat = cv2.applyColorMap(cv2.normalize(capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cm_choice)
        else:
            heat = np.zeros((h, w, 3), dtype=np.uint8)
        combined_heatmap[y:y+h, x:x+w] = heat
        combined_mask[y:y+h, x:x+w] = True
    overlay_bgr = first_frame.copy()
    overlay_bgr[combined_mask] = cv2.addWeighted(overlay_bgr[combined_mask], 0.6, combined_heatmap[combined_mask], 0.6, 0)
    if add_scale_bar and mpp > 0:
        px_len = int(scale_m / mpp)
        px_len = max(px_len, 1)
        pad = 24
        x0 = pad
        y0 = h_full - pad
        cv2.line(overlay_bgr, (x0, y0), (x0 + px_len, y0), (0, 0, 0), 4)
        cv2.putText(overlay_bgr, f"{scale_m:.1f} m", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,0), 2, cv2.LINE_AA)
    return overlay_bgr

def compute_measured_vertical_profile(avg_maps, rois, mpp):
    rows = {}
    for (x, y, w, h), amap in zip(rois, avg_maps):
        if amap.size == 0: 
            continue
        for r in range(h):
            y_abs = y + r
            if y_abs not in rows:
                rows[y_abs] = []
            rows[y_abs].append(float(np.nanmean(amap[r, :])))
    if not rows:
        return np.array([]), np.array([])
    y_sorted = sorted(rows.keys())
    vel_avg = np.array([np.nanmean(rows[yy]) for yy in y_sorted], dtype=float)
    depth_m = np.array(y_sorted, dtype=float) * float(mpp)
    return depth_m, vel_avg

overlay_initial = build_overlay_from_avg_maps("JET", global_cap, add_scale_bar=False)

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
})

def cv_bgr_to_qpix(img_bgr):
    h, w, ch = img_bgr.shape
    bytes_per_line = ch * w
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def inside_any_roi(ix, iy, rois):
    for ri, (rx, ry, rw, rh) in enumerate(rois):
        if rx <= ix < rx + rw and ry <= iy < ry + rh:
            return ri, ix - rx, iy - ry
    return None, None, None

class ColorbarCanvas(FigureCanvas):
    def __init__(self, vmax, cmap_name="JET"):
        self.vmax = vmax if vmax and vmax > 0 else 1.0
        self.cmap_name = cmap_name
        fig = Figure(figsize=(1.3, 4.2), dpi=120)
        super().__init__(fig)
        self.ax = fig.add_axes([0.25, 0.07, 0.55, 0.9])
        self._draw_bar()

    def _mpl_cmap(self):
        name = self.cmap_name.upper()
        if name == "TURBO":
            return mpl.colormaps.get_cmap("turbo")
        if name == "VIRIDIS":
            return mpl.colormaps.get_cmap("viridis")
        return mpl.colormaps.get_cmap("jet")

    def _draw_bar(self):
        self.ax.clear()
        norm = mpl.colors.Normalize(vmin=0.0, vmax=self.vmax)
        cb = mpl.colorbar.ColorbarBase(self.ax, cmap=self._mpl_cmap(), norm=norm, orientation='vertical')
        cb.set_label("Velocity (m/s)", fontweight="bold", fontsize=16)
        self.ax.tick_params(axis='y', labelsize=14)
        for l in self.ax.get_yticklabels():
            l.set_fontweight("bold")
        self.draw_idle()

    def update_bar(self, vmax, cmap_name):
        self.vmax = vmax if vmax and vmax > 0 else 1.0
        self.cmap_name = cmap_name
        self._draw_bar()

class LivePlot(FigureCanvas):
    def __init__(self, parent=None, fps=25.0, mpp=0.039):
        self.display_mode = "both"
        self.mode = "timeseries"
        self.fps = float(fps)
        self.mpp = float(mpp)
        fig = Figure(figsize=(6.2, 4.2), dpi=110)
        super().__init__(fig)
        self.ax_main = None
        self.ax_hist = None
        self.line_main = None
        self._rebuild_layout()

    def set_mode(self, mode):
        if mode not in ("timeseries", "hprofile", "vprofile"):
            return
        if self.mode != mode:
            self.mode = mode
            self._init_axes_labels()

    def set_display_mode(self, m):
        if m not in ("both", "main_only", "hist_only"):
            return
        if self.display_mode != m:
            self.display_mode = m
            self._rebuild_layout()

    def _rebuild_layout(self):
        self.figure.clf()
        if self.display_mode == "both":
            self.ax_main = self.figure.add_subplot(211)
            self.ax_hist = self.figure.add_subplot(212)
        elif self.display_mode == "main_only":
            self.ax_main = self.figure.add_subplot(111)
            self.ax_hist = None
        else:
            self.ax_main = None
            self.ax_hist = self.figure.add_subplot(111)
        self._init_axes_labels()
        self.figure.tight_layout()
        self.draw_idle()

    def _init_axes_labels(self):
        if self.ax_main is not None:
            self.ax_main.clear()
            if self.mode == "timeseries":
                self.ax_main.set_title("Velocity at point", fontweight="bold")
                self.ax_main.set_xlabel("Time (s)")
                self.ax_main.set_ylabel("Velocity (m/s)")
                (self.line_main,) = self.ax_main.plot([], [], lw=2.4, label="Measured")
                self.ax_main.grid(True)
            elif self.mode == "hprofile":
                self.ax_main.set_title("Horizontal velocity profile", fontweight="bold")
                self.ax_main.set_xlabel("Cross-stream distance (m)")
                self.ax_main.set_ylabel("Velocity (m/s)")
                (self.line_main,) = self.ax_main.plot([], [], lw=2.4, label="Measured")
                self.ax_main.grid(True)
            else:
                self.ax_main.set_title("Vertical velocity profile", fontweight="bold")
                self.ax_main.set_xlabel("Velocity (m/s)")
                self.ax_main.set_ylabel("Depth (m)")
                (self.line_main,) = self.ax_main.plot([], [], lw=2.4, label="Measured")
                self.ax_main.grid(True)
                self.ax_main.invert_yaxis()
            self.ax_main.tick_params(axis='both', labelsize=14)
            self.ax_main.legend(loc="best", fontsize=14)

        if self.ax_hist is not None:
            self.ax_hist.clear()
            self.ax_hist.set_title("Histogram of velocities", fontweight="bold")
            self.ax_hist.set_xlabel("Velocity (m/s)")
            self.ax_hist.set_ylabel("Count")
            self.ax_hist.tick_params(axis='both', labelsize=14)
        self.figure.tight_layout()

    def _plot_hist(self, values):
        if self.ax_hist is None:
            return
        self.ax_hist.clear()
        self.ax_hist.set_title("Histogram of velocities", fontweight="bold")
        self.ax_hist.set_xlabel("Velocity (m/s)")
        self.ax_hist.set_ylabel("Count")
        if values:
            vals = np.asarray(values, dtype=float)
            counts, bin_edges = np.histogram(vals, bins="auto")
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width = (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
            self.ax_hist.bar(centers, counts, width=width)
        self.ax_hist.tick_params(axis='both', labelsize=14)
        self.figure.tight_layout()
        self.draw_idle()

    def update_series(self, vel_series_mps):
        self.set_mode("timeseries")
        ys = list(map(float, vel_series_mps))
        if self.ax_main is not None:
            xs = np.arange(len(ys), dtype=float) / max(1.0, self.fps)
            self.line_main.set_data(xs, ys)
            self.ax_main.set_xlim(0, xs[-1] if len(xs) else 1)
            ymax = max(1e-6, (max(ys) if ys else 1.0))
            self.ax_main.set_ylim(0, ymax * 1.2)
        self._plot_hist(ys)
        self.draw_idle()

    def update_hprofile(self, values_1d_mps, pixel_len):
        self.set_mode("hprofile")
        ys = list(map(float, values_1d_mps)) if values_1d_mps is not None else []
        if self.ax_main is not None:
            n = len(ys)
            xs = (np.arange(pixel_len, dtype=float)[:n] * self.mpp) if n else []
            self.line_main.set_data(xs, ys)
            self.ax_main.set_xlim(0, xs[-1] if len(xs) else 1)
            ymax = max(1e-6, (max(ys) if ys else 1.0))
            self.ax_main.set_ylim(0, ymax * 1.2)
        self._plot_hist(ys)
        self.draw_idle()

    def update_vprofile(self, measured_values_mps, pixel_len):
        self.set_mode("vprofile")
        xs_speed = list(map(float, measured_values_mps)) if measured_values_mps is not None else []
        if self.ax_main is not None:
            n = len(xs_speed)
            ys_depth = (np.arange(pixel_len, dtype=float)[:n] * self.mpp) if n else []
            self.line_main.set_data(xs_speed, ys_depth)
            xmax = max(1e-6, (max(xs_speed) if xs_speed else 1.0))
            self.ax_main.set_xlim(0, xmax * 1.2)
            if len(ys_depth):
                self.ax_main.set_ylim(ys_depth[-1], 0)
        self._plot_hist(xs_speed)
        self.draw_idle()

class ValidationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Velocity vs Depth Comparison")
        self.resize(900, 700)

        central = QtWidgets.QWidget(self)
        layout = QVBoxLayout(central)
        self.setLayout(layout)

        fig = Figure(figsize=(9, 6), dpi=110)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        ax = fig.add_subplot(111)
        ax.set_title("Velocity vs Depth Comparison", fontsize=16, fontweight="bold")
        ax.set_ylabel("Velocity (m/s)", fontsize=14)
        ax.set_xlabel("Depth (m)", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        
        validation_data = pd.read_csv(validation_path)
        validation_depth = validation_data['Depth m']
        validation_velocity = validation_data['Velocity m/s']

        output_data = pd.read_csv(output_path)
        output_depth = output_data['Depth m']
        output_velocity = output_data['Velocity m/s']

        ax.plot(validation_depth, validation_velocity, 'r--o', linewidth=2.0, markersize=6, label="Validation Data")
        ax.plot(output_depth, output_velocity, 'b--o', linewidth=2.0, markersize=6, label="Output Data")
        ax.legend(fontsize=12)
        fig.tight_layout()
        self.canvas.draw_idle()



class ResultsViewer(QMainWindow):
    def __init__(self, overlay_bgr, avg_maps, vel_history, rois, fps, mpp, vmax, cmap_name="JET",
                 validation_depth=None, validation_vel=None,
                 measured_profile_depth=None, measured_profile_vel=None):
        super().__init__()
        self.setWindowTitle("Spillway Results — Post-Processing Viewer")
        self.resize(1400, 900)

        self.base_overlay = overlay_bgr.copy()
        self.overlay_bgr = overlay_bgr.copy()
        self.overlay_h, self.overlay_w = overlay_bgr.shape[:2]
        self.avg_maps = avg_maps
        self.vel_history = vel_history
        self.rois = rois
        self.fps = float(fps)
        self.mpp = float(mpp)
        self.vmax = float(vmax) if vmax and vmax > 0 else 1.0
        self.cmap_name = cmap_name

        self.have_hover = False
        self.hover_x = -1
        self.hover_y = -1
        self.pin = None
        self.profile_sel = None

        self.validation_depth = validation_depth
        self.validation_vel = validation_vel
        self.measured_profile_depth = measured_profile_depth
        self.measured_profile_vel = measured_profile_vel

        self._build_ui()

        self.base_pix = cv_bgr_to_qpix(self.overlay_bgr)
        self.metric_label.setPixmap(self.base_pix.scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left_col = QVBoxLayout()
        self.metric_label = QLabel()
        self.metric_label.setAlignment(Qt.AlignCenter)
        self.metric_label.setMouseTracking(True)
        self.metric_label.installEventFilter(self)
        left_col.addWidget(self.metric_label, stretch=5)
        left_col.addWidget(self._hud_widget(), stretch=0)

        self.plot = LivePlot(fps=self.fps, mpp=self.mpp)

        right_col = QVBoxLayout()
        settings_box = QGroupBox("Settings")
        settings_box.setFont(QFont("Segoe UI", 15, QFont.Bold))
        form = QFormLayout(settings_box)

        self.combo_hover = QComboBox()
        self.combo_hover.addItems(["Point", "Vertical", "Horizontal"])
        self.combo_hover.currentTextChanged.connect(self._set_hover_mode)
        self.combo_hover.setFont(QFont("Segoe UI", 14))

        self.cb_crosshair = QCheckBox("Show crosshair in Point mode")
        self.cb_crosshair.setChecked(True)
        self.cb_crosshair.setFont(QFont("Segoe UI", 14))

        self.combo_plot = QComboBox()
        self.combo_plot.addItems(["Both", "Velocity/profile only", "Histogram only"])
        self.combo_plot.currentTextChanged.connect(self._set_plot_content)
        self.combo_plot.setFont(QFont("Segoe UI", 14))

        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["JET", "TURBO", "VIRIDIS"])
        self.combo_cmap.setCurrentText(self.cmap_name)
        self.combo_cmap.currentTextChanged.connect(self._change_cmap)
        self.combo_cmap.setFont(QFont("Segoe UI", 14))

        self.cb_scalebar = QCheckBox("Show 1 m scale bar")
        self.cb_scalebar.setChecked(False)
        self.cb_scalebar.stateChanged.connect(self._toggle_scalebar)
        self.cb_scalebar.setFont(QFont("Segoe UI", 14))

        self.btn_validation = QPushButton("View Validation Graph")
        self.btn_validation.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.btn_validation.clicked.connect(self._open_validation_dialog)

        form.addRow(self._label_bold("Hover mode:"), self.combo_hover)
        form.addRow(self.cb_crosshair)
        form.addRow(self._label_bold("Plot content:"), self.combo_plot)
        form.addRow(self._label_bold("Colormap:"), self.combo_cmap)
        form.addRow(self.cb_scalebar)
        form.addRow(self.btn_validation)

        cb_group = QGroupBox("Legend (m/s)")
        cb_group.setFont(QFont("Segoe UI", 15, QFont.Bold))
        cb_layout = QVBoxLayout(cb_group)
        self.colorbar = ColorbarCanvas(self.vmax, self.cmap_name)
        cb_layout.addWidget(self.colorbar)

        right_col.addWidget(settings_box, stretch=0)
        right_col.addWidget(cb_group, stretch=1)

        layout.addLayout(left_col, stretch=4)
        layout.addWidget(self.plot, stretch=4)
        layout.addLayout(right_col, stretch=2)

    def _label_bold(self, text):
        lab = QLabel(text)
        lab.setFont(QFont("Segoe UI", 15, QFont.Bold))
        return lab

    def _hud_widget(self):
        box = QGroupBox("HUD")
        box.setFont(QFont("Segoe UI", 15, QFont.Bold))
        self.hud_hover = QLabel("Hover: --")
        self.hud_locked = QLabel("Pinned: --")
        self.hud_hint = QLabel("Point: click for time series (t in s). Vertical/Horizontal: click for profile (m).")
        for lab in (self.hud_hover, self.hud_locked, self.hud_hint):
            lab.setFont(QFont("Segoe UI", 15))
        lay = QVBoxLayout(box)
        lay.addWidget(self.hud_hover)
        lay.addWidget(self.hud_locked)
        lay.addWidget(self.hud_hint)
        return box

    def _set_hover_mode(self, text):
        self.profile_sel = None
        if text == "Point":
            self.plot.set_mode("timeseries")
        elif text == "Vertical":
            self.plot.set_mode("vprofile")
        else:
            self.plot.set_mode("hprofile")
        self._redraw_overlay()

    def _set_plot_content(self, text):
        if text.startswith("Both"):
            self.plot.set_display_mode("both")
        elif text.startswith("Velocity"):
            self.plot.set_display_mode("main_only")
        else:
            self.plot.set_display_mode("hist_only")
        self._refresh_plot_after_mode_change()

    def _change_cmap(self, name):
        self.cmap_name = name
        self._rebuild_overlay_from_controls()
        self.colorbar.update_bar(self.vmax, self.cmap_name)

    def _toggle_scalebar(self, state):
        self._rebuild_overlay_from_controls()

    def _rebuild_overlay_from_controls(self):
        show_bar = self.cb_scalebar.isChecked()
        self.overlay_bgr = build_overlay_from_avg_maps(self.cmap_name, self.vmax, add_scale_bar=show_bar, scale_m=1.0)
        self.base_pix = cv_bgr_to_qpix(self.overlay_bgr)
        self.metric_label.setPixmap(self.base_pix.scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_plot_after_mode_change(self):
        if self.pin is not None and self.combo_hover.currentText() == "Point":
            ix, iy = self.pin
            roi_idx, lx, ly = inside_any_roi(ix, iy, self.rois)
            if roi_idx is not None:
                self._plot_time_series(roi_idx, lx, ly)
        elif self.profile_sel is not None:
            axis, idx = self.profile_sel
            if axis == "V":
                x = int(np.clip(idx, 0, self.overlay_w-1))
                roi_idx, lx, _ = inside_any_roi(x, 0, self.rois)
                if roi_idx is not None:
                    self._plot_vertical_profile(roi_idx, lx)
            else:
                y = int(np.clip(idx, 0, self.overlay_h-1))
                roi_idx, _, ly = inside_any_roi(0, y, self.rois)
                if roi_idx is not None:
                    self._plot_horizontal_profile(roi_idx, ly)
        self._redraw_overlay()

    def eventFilter(self, obj, event):
        if obj is self.metric_label:
            if event.type() == QtCore.QEvent.MouseMove:
                self.have_hover = True
                self.hover_x, self.hover_y = event.pos().x(), event.pos().y()
                self._update_hover()
            elif event.type() == QtCore.QEvent.Leave:
                self.have_hover = False
                self.hud_hover.setText("Hover: --")
                self._redraw_overlay()
            elif event.type() == QtCore.QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._handle_click()
        return super().eventFilter(obj, event)

    def _widget_to_image_coords(self, label: QLabel, x, y):
        pm = self.base_pix
        img_w = pm.width()
        img_h = pm.height()
        lab_w = label.width()
        lab_h = label.height()
        scale = min(lab_w / img_w, lab_h / img_h)
        draw_w, draw_h = int(img_w * scale), int(img_h * scale)
        off_x = (lab_w - draw_w) // 2
        off_y = (lab_h - draw_h) // 2
        ix = int((x - off_x) / scale)
        iy = int((y - off_y) / scale)
        ix = np.clip(ix, 0, img_w - 1)
        iy = np.clip(iy, 0, img_h - 1)
        return int(ix), int(iy)

    def _update_hover(self):
        ix, iy = self._widget_to_image_coords(self.metric_label, self.hover_x, self.hover_y)
        roi_idx, lx, ly = inside_any_roi(ix, iy, self.rois)
        disp = self.overlay_bgr.copy()
        x_m = ix * self.mpp
        y_m = iy * self.mpp
        mode = self.combo_hover.currentText()

        if mode == "Point":
            cv2.circle(disp, (ix, iy), 8, (0,255,0), 2)
            if self.cb_crosshair.isChecked():
                cv2.line(disp, (ix, 0), (ix, self.overlay_h-1), (255,255,255), 1)
                cv2.line(disp, (0, iy), (self.overlay_w-1, iy), (255,255,255), 1)
        elif mode == "Vertical":
            cv2.line(disp, (ix, 0), (ix, self.overlay_h-1), (0,255,0), 3)
        else:
            cv2.line(disp, (0, iy), (self.overlay_w-1, iy), (0,255,0), 3)

        if roi_idx is not None:
            amap = self.avg_maps[roi_idx]
            H, W = amap.shape
            if 0 <= ly < H and 0 <= lx < W:
                v = float(amap[ly, lx])
                self.hud_hover.setText(f"Hover: ROI {roi_idx+1} @ ({x_m:.3f} m, {y_m:.3f} m) = {v:.3f} m/s")
            else:
                self.hud_hover.setText(f"Hover: ({x_m:.3f} m, {y_m:.3f} m) — outside ROI grid")
        else:
            self.hud_hover.setText(f"Hover: ({x_m:.3f} m, {y_m:.3f} m) — outside ROIs")

        pix = cv_bgr_to_qpix(disp)
        self.metric_label.setPixmap(pix.scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _handle_click(self):
        ix, iy = self._widget_to_image_coords(self.metric_label, self.hover_x, self.hover_y)
        roi_idx, lx, ly = inside_any_roi(ix, iy, self.rois)
        if roi_idx is None:
            return
        disp = self.overlay_bgr.copy()
        mode = self.combo_hover.currentText()
        if mode == "Point":
            self.pin = (ix, iy)
            x_m, y_m = ix * self.mpp, iy * self.mpp
            self.hud_locked.setText(f"Pinned: ROI {roi_idx+1} @ ({x_m:.3f} m, {y_m:.3f} m)")
            self._plot_time_series(roi_idx, lx, ly)
            cv2.drawMarker(disp, (ix, iy), (0,0,255), cv2.MARKER_CROSS, 18, 3)
        elif mode == "Vertical":
            self.profile_sel = ("V", ix)
            self.hud_locked.setText(f"Pinned: Vertical profile at x={ix*self.mpp:.3f} m")
            self._plot_vertical_profile(roi_idx, lx)
            cv2.line(disp, (ix, 0), (ix, self.overlay_h-1), (0,255,0), 3)
        else:
            self.profile_sel = ("H", iy)
            self.hud_locked.setText(f"Pinned: Horizontal profile at y={iy*self.mpp:.3f} m")
            self._plot_horizontal_profile(roi_idx, ly)
            cv2.line(disp, (0, iy), (self.overlay_w-1, iy), (0,255,0), 3)
        pix = cv_bgr_to_qpix(disp)
        self.metric_label.setPixmap(pix.scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _redraw_overlay(self):
        disp = self.overlay_bgr.copy()
        if self.profile_sel is not None:
            axis, idx = self.profile_sel
            if axis == "V":
                x = int(np.clip(idx, 0, self.overlay_w-1))
                cv2.line(disp, (x, 0), (x, self.overlay_h-1), (0,255,0), 3)
            else:
                y = int(np.clip(idx, 0, self.overlay_h-1))
                cv2.line(disp, (0, y), (self.overlay_w-1, y), (0,255,0), 3)
        if self.pin is not None:
            px, py = self.pin
            cv2.drawMarker(disp, (px, py), (0,0,255), cv2.MARKER_CROSS, 18, 3)
        pix = cv_bgr_to_qpix(disp)
        self.metric_label.setPixmap(pix.scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _plot_time_series(self, roi_idx, lx, ly):
        hist = self.vel_history[roi_idx]
        if len(hist) == 0:
            self.plot.update_series([])
            return
        H, W = hist[0].shape
        if not (0 <= ly < H and 0 <= lx < W):
            self.plot.update_series([])
            return
        ts = np.array([f[ly, lx] for f in hist], dtype=np.float32)
        self.plot.update_series(list(ts))

    def _plot_horizontal_profile(self, roi_idx, ly):
        amap = self.avg_maps[roi_idx]
        if amap.size == 0:
            self.plot.update_hprofile([], 0)
            return
        H, W = amap.shape
        ly = int(np.clip(ly, 0, H-1))
        self.plot.update_hprofile(amap[ly, :], W)

    def _plot_vertical_profile(self, roi_idx, lx):
        if len(self.avg_maps) < 2:
            self.plot.update_vprofile([], 0)
            return

        top = self.avg_maps[0]
        bot = self.avg_maps[1]

        if top.size == 0 or bot.size == 0:
            self.plot.update_vprofile([], 0)
            return

        H1, W1 = top.shape
        H2, W2 = bot.shape

        if hasattr(self.plot, "display_width") and self.plot.display_width:
            f = np.clip(lx / float(self.plot.display_width), 0.0, 1.0)
            lx1 = int(np.clip(round(f * (W1 - 1)), 0, W1 - 1))
            lx2 = int(np.clip(round(f * (W2 - 1)), 0, W2 - 1))
        else:
            lx1 = int(np.clip(lx, 0, W1 - 1))
            lx2 = int(np.clip(lx, 0, W2 - 1))

        col1 = top[:, lx1]    # top ROI
        col2 = bot[:, lx2]    # bottom ROI
        profile = np.concatenate([col1, col2], axis=0)
        total_h = H1 + H2

        self.plot.update_vprofile(profile, total_h)

    def _open_validation_dialog(self):
        if (self.validation_depth is not None and len(self.validation_depth)
            and self.measured_profile_depth is not None and len(self.measured_profile_depth)):
            dlg = ValidationDialog()
            dlg.exec_()
        else:
            QtWidgets.QMessageBox.information(self, "Validation", "No validation data available.")

ideal_depth_m, ideal_velocity = None, None
if os.path.exists(validation_path):
    try:
        ds, vs = [], []
        with open(validation_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                d = row.get("Depth m")
                v = row.get("Velocity m/s")
                if d is None or v is None:
                    continue
                ds.append(float(d)); vs.append(float(v))
        if len(ds) and len(vs):
            ideal_depth_m = np.array(ds, dtype=float)
            ideal_velocity = np.array(vs, dtype=float)
    except Exception as e:
        print("Validation file read error:", e)

meas_depth_m, meas_vel_mps = compute_measured_vertical_profile(avg_maps, rois, mpp)

app = QApplication(sys.argv)
win = ResultsViewer(
    overlay_bgr=overlay_initial,
    avg_maps=avg_maps,
    vel_history=vel_history,
    rois=rois,
    fps=fps,
    mpp=mpp,
    vmax=global_cap if global_cap>0 else 1.0,
    cmap_name="JET",
    validation_depth=ideal_depth_m,
    validation_vel=ideal_velocity,
    measured_profile_depth=meas_depth_m,
    measured_profile_vel=meas_vel_mps
)
win.show()
sys.exit(app.exec_())
