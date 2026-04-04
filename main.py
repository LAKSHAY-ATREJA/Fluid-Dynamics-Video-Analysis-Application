"""
main.py — Fluid dynamics velocity analysis from video.

Combines dense optical flow (Farneback) with Shape-from-Shading (SfS) depth
estimation to produce spatially calibrated velocity maps, vertical/horizontal
flow profiles, time-series pixel inspection, and a validation comparison
against reference ideal-flow data.

Usage
-----
    python main.py --video /path/to/video.MOV [options]

See ``python main.py --help`` for all options.
"""

import sys
import csv
import os
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl

from preprocessing import (
    rotate_image,
    resize_to_screen,
    apply_clahe,
    apply_gamma_correction,
    point_to_point,
    select_gamma_for_roi,
)
from SFSImplementation import forward_gradients, run_sfs_on_frame

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QPushButton,
    QDialog,
)
from PyQt5.QtGui import QPixmap, QImage, QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ===========================================================================
# Argument parsing
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fluid dynamics velocity analysis from video using optical flow and SfS."
    )
    parser.add_argument(
        "--video", default="IMGP9471.MOV",
        help="Path to the input video file (default: IMGP9471.MOV)",
    )
    parser.add_argument(
        "--validation", default="ideal.csv",
        help="Path to the ideal/validation CSV (default: ideal.csv)",
    )
    parser.add_argument(
        "--output", default="velocity_output.csv",
        help="Path for the output velocity CSV (default: velocity_output.csv)",
    )
    parser.add_argument(
        "--width-m", type=float, default=12.25,
        help="Known calibration width in metres (default: 12.25)",
    )
    parser.add_argument(
        "--rotation", type=int, default=270,
        choices=[0, 90, 180, 270],
        help="Frame rotation angle in degrees (default: 270)",
    )
    return parser.parse_args()


# ===========================================================================
# Camera perspective correction
# ===========================================================================

# Piecewise-linear angular interpolation derived from known dam geometry.
# Each pair maps a pixel row position to the view angle in degrees.
HEIGHT_BREAKS = [544, 555, 579, 613, 648, 691, 731, 768, 813]
HEIGHT_SCALES = [25.84, 31.67, 35.84, 39.09, 42.89, 45.86, 48.28, 50.30, 51.34]


# ===========================================================================
# Qt helpers
# ===========================================================================

def cv_bgr_to_qpix(img_bgr):
    """Convert a BGR ndarray to a QPixmap."""
    h, w, ch = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def inside_any_roi(ix, iy, rois):
    """
    Test whether pixel ``(ix, iy)`` falls inside any of *rois*.

    Returns ``(roi_index, local_x, local_y)`` if found, or
    ``(None, None, None)`` otherwise.
    """
    for ri, (rx, ry, rw, rh) in enumerate(rois):
        if rx <= ix < rx + rw and ry <= iy < ry + rh:
            return ri, ix - rx, iy - ry
    return None, None, None


# ===========================================================================
# Qt widgets
# ===========================================================================

class ColorbarCanvas(FigureCanvas):
    """Vertical velocity colorbar rendered via Matplotlib."""

    def __init__(self, vmax, cmap_name="JET"):
        self.vmax = max(1e-6, float(vmax) if vmax else 1.0)
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
        cb = mpl.colorbar.ColorbarBase(
            self.ax, cmap=self._mpl_cmap(), norm=norm, orientation="vertical"
        )
        cb.set_label("Velocity (m/s)", fontweight="bold", fontsize=16)
        self.ax.tick_params(axis="y", labelsize=14)
        for lbl in self.ax.get_yticklabels():
            lbl.set_fontweight("bold")
        self.draw_idle()

    def update_bar(self, vmax, cmap_name):
        self.vmax = max(1e-6, float(vmax) if vmax else 1.0)
        self.cmap_name = cmap_name
        self._draw_bar()


class LivePlot(FigureCanvas):
    """
    Matplotlib canvas that renders one of three plot modes:
    - ``timeseries``: velocity at a clicked pixel over time.
    - ``hprofile``: average velocity across a horizontal scan line.
    - ``vprofile``: average velocity across a vertical scan line.

    A velocity-distribution histogram occupies the lower panel when
    ``display_mode`` is ``"both"`` (default).
    """

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
            elif self.mode == "hprofile":
                self.ax_main.set_title("Horizontal velocity profile", fontweight="bold")
                self.ax_main.set_xlabel("Cross-stream distance (m)")
                self.ax_main.set_ylabel("Velocity (m/s)")
            else:
                self.ax_main.set_title("Vertical velocity profile", fontweight="bold")
                self.ax_main.set_xlabel("Velocity (m/s)")
                self.ax_main.set_ylabel("Depth (m)")
                self.ax_main.invert_yaxis()
            (self.line_main,) = self.ax_main.plot([], [], lw=2.4, label="Measured")
            self.ax_main.grid(True)
            self.ax_main.tick_params(axis="both", labelsize=14)
            self.ax_main.legend(loc="best", fontsize=14)

        if self.ax_hist is not None:
            self.ax_hist.clear()
            self.ax_hist.set_title("Histogram of velocities", fontweight="bold")
            self.ax_hist.set_xlabel("Velocity (m/s)")
            self.ax_hist.set_ylabel("Count")
            self.ax_hist.tick_params(axis="both", labelsize=14)
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
        self.ax_hist.tick_params(axis="both", labelsize=14)
        self.figure.tight_layout()
        self.draw_idle()

    def update_series(self, vel_series_mps):
        self.set_mode("timeseries")
        ys = list(map(float, vel_series_mps))
        if self.ax_main is not None:
            xs = np.arange(len(ys), dtype=float) / max(1.0, self.fps)
            self.line_main.set_data(xs, ys)
            self.ax_main.set_xlim(0, xs[-1] if len(xs) else 1)
            ymax = max(1e-6, max(ys) if ys else 1.0)
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
            ymax = max(1e-6, max(ys) if ys else 1.0)
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
            xmax = max(1e-6, max(xs_speed) if xs_speed else 1.0)
            self.ax_main.set_xlim(0, xmax * 1.2)
            if len(ys_depth):
                self.ax_main.set_ylim(ys_depth[-1], 0)
        self._plot_hist(xs_speed)
        self.draw_idle()


class ValidationDialog(QDialog):
    """
    Modal dialog showing measured velocity vs depth overlaid on the ideal
    (validation) dataset.  Both datasets are read from CSV files that are
    confirmed to exist before this dialog is opened.
    """

    def __init__(self, validation_path, output_path):
        super().__init__()
        self.setWindowTitle("Velocity vs Depth Comparison")
        self.resize(900, 700)

        layout = QVBoxLayout(self)

        fig = Figure(figsize=(9, 6), dpi=110)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.set_title("Velocity vs Depth Comparison", fontsize=16, fontweight="bold")
        ax.set_ylabel("Velocity (m/s)", fontsize=14)
        ax.set_xlabel("Depth (m)", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)

        try:
            val_df = pd.read_csv(validation_path)
            ax.plot(
                val_df["Depth m"], val_df["Velocity m/s"],
                "r--o", linewidth=2.0, markersize=6, label="Validation (ideal)",
            )
        except Exception as exc:
            ax.set_title(f"Could not load validation data: {exc}", fontsize=12)

        try:
            out_df = pd.read_csv(output_path)
            ax.plot(
                out_df["Depth m"], out_df["Velocity m/s"],
                "b--o", linewidth=2.0, markersize=6, label="Measured",
            )
        except Exception as exc:
            print(f"Warning: could not load output data for validation plot: {exc}",
                  file=sys.stderr)

        ax.legend(fontsize=12)
        fig.tight_layout()
        canvas.draw_idle()


class ResultsViewer(QMainWindow):
    """
    Interactive Qt results viewer launched after the optical-flow processing
    loop terminates.

    Provides:
    - Heatmap overlay with switchable colormaps (JET / TURBO / VIRIDIS)
    - Hover-to-inspect: point velocity, vertical profile, horizontal profile
    - Click-to-pin: locks the displayed plot to the selected pixel/line
    - Velocity histogram below each profile plot
    - Optional 1 m scale bar overlay
    - Validation comparison dialog (if ideal.csv and output CSV are available)
    """

    def __init__(
        self,
        overlay_bgr,
        avg_maps,
        vel_history,
        rois,
        fps,
        mpp,
        vmax,
        cmap_name="JET",
        validation_depth=None,
        validation_vel=None,
        measured_profile_depth=None,
        measured_profile_vel=None,
        validation_path="ideal.csv",
        output_path="velocity_output.csv",
    ):
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
        self.validation_path = validation_path
        self.output_path = output_path

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
        self.metric_label.setPixmap(
            self.base_pix.scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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
        self.hud_hint = QLabel(
            "Point: click for time series.  Vertical/Horizontal: click for profile."
        )
        for lab in (self.hud_hover, self.hud_locked, self.hud_hint):
            lab.setFont(QFont("Segoe UI", 15))
        lay = QVBoxLayout(box)
        lay.addWidget(self.hud_hover)
        lay.addWidget(self.hud_locked)
        lay.addWidget(self.hud_hint)
        return box

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

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

    def _toggle_scalebar(self, _state):
        self._rebuild_overlay_from_controls()

    def _rebuild_overlay_from_controls(self):
        show_bar = self.cb_scalebar.isChecked()
        self.overlay_bgr = self._build_overlay(self.cmap_name, self.vmax, add_scale_bar=show_bar)
        self.base_pix = cv_bgr_to_qpix(self.overlay_bgr)
        self.metric_label.setPixmap(
            self.base_pix.scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _refresh_plot_after_mode_change(self):
        if self.pin is not None and self.combo_hover.currentText() == "Point":
            ix, iy = self.pin
            roi_idx, lx, ly = inside_any_roi(ix, iy, self.rois)
            if roi_idx is not None:
                self._plot_time_series(roi_idx, lx, ly)
        elif self.profile_sel is not None:
            axis, idx = self.profile_sel
            if axis == "V":
                x = int(np.clip(idx, 0, self.overlay_w - 1))
                roi_idx, lx, _ = inside_any_roi(x, 0, self.rois)
                if roi_idx is not None:
                    self._plot_vertical_profile(roi_idx, lx)
            else:
                y = int(np.clip(idx, 0, self.overlay_h - 1))
                roi_idx, _, ly = inside_any_roi(0, y, self.rois)
                if roi_idx is not None:
                    self._plot_horizontal_profile(roi_idx, ly)
        self._redraw_overlay()

    # ------------------------------------------------------------------
    # Overlay construction
    # ------------------------------------------------------------------

    def _build_overlay(self, colormap_name="JET", cap_val=None, add_scale_bar=False, scale_m=1.0):
        """Render velocity heatmap overlay on top of the first frame."""
        cmap_table = {
            "JET": cv2.COLORMAP_JET,
            "TURBO": cv2.COLORMAP_TURBO,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        }
        cm_choice = cmap_table.get(colormap_name.upper(), cv2.COLORMAP_JET)

        combined_heatmap = np.zeros((self.overlay_h, self.overlay_w, 3), dtype=np.uint8)
        combined_mask = np.zeros((self.overlay_h, self.overlay_w), dtype=bool)

        for (rx, ry, rw, rh), amap in zip(self.rois, self.avg_maps):
            if amap.size == 0:
                continue
            cap = cap_val if (cap_val is not None and cap_val > 0) else max(1e-6, float(np.max(amap)))
            capped = np.clip(amap, 0, cap)
            norm = cv2.normalize(capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heat = cv2.applyColorMap(norm, cm_choice)
            combined_heatmap[ry : ry + rh, rx : rx + rw] = heat
            combined_mask[ry : ry + rh, rx : rx + rw] = True

        out = self.base_overlay.copy()
        out[combined_mask] = cv2.addWeighted(
            out[combined_mask], 0.6, combined_heatmap[combined_mask], 0.6, 0
        )

        if add_scale_bar and self.mpp > 0:
            px_len = max(1, int(scale_m / self.mpp))
            pad = 24
            x0, y0 = pad, self.overlay_h - pad
            cv2.line(out, (x0, y0), (x0 + px_len, y0), (0, 0, 0), 4)
            cv2.putText(
                out, f"{scale_m:.1f} m", (x0, y0 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

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
        """Map widget-space coordinates to image-space coordinates."""
        pm = self.base_pix
        img_w, img_h = pm.width(), pm.height()
        lab_w, lab_h = label.width(), label.height()
        scale = min(lab_w / max(img_w, 1), lab_h / max(img_h, 1))
        draw_w, draw_h = int(img_w * scale), int(img_h * scale)
        off_x = (lab_w - draw_w) // 2
        off_y = (lab_h - draw_h) // 2
        ix = int((x - off_x) / max(scale, 1e-8))
        iy = int((y - off_y) / max(scale, 1e-8))
        ix = int(np.clip(ix, 0, img_w - 1))
        iy = int(np.clip(iy, 0, img_h - 1))
        return ix, iy

    def _update_hover(self):
        ix, iy = self._widget_to_image_coords(self.metric_label, self.hover_x, self.hover_y)
        roi_idx, lx, ly = inside_any_roi(ix, iy, self.rois)
        disp = self.overlay_bgr.copy()
        x_m = ix * self.mpp
        y_m = iy * self.mpp
        mode = self.combo_hover.currentText()

        if mode == "Point":
            cv2.circle(disp, (ix, iy), 8, (0, 255, 0), 2)
            if self.cb_crosshair.isChecked():
                cv2.line(disp, (ix, 0), (ix, self.overlay_h - 1), (255, 255, 255), 1)
                cv2.line(disp, (0, iy), (self.overlay_w - 1, iy), (255, 255, 255), 1)
        elif mode == "Vertical":
            cv2.line(disp, (ix, 0), (ix, self.overlay_h - 1), (0, 255, 0), 3)
        else:
            cv2.line(disp, (0, iy), (self.overlay_w - 1, iy), (0, 255, 0), 3)

        if roi_idx is not None:
            amap = self.avg_maps[roi_idx]
            h_amap, w_amap = amap.shape
            if 0 <= ly < h_amap and 0 <= lx < w_amap:
                v = float(amap[ly, lx])
                self.hud_hover.setText(
                    f"Hover: ROI {roi_idx + 1} @ ({x_m:.3f} m, {y_m:.3f} m) = {v:.3f} m/s"
                )
            else:
                self.hud_hover.setText(f"Hover: ({x_m:.3f} m, {y_m:.3f} m) — outside ROI grid")
        else:
            self.hud_hover.setText(f"Hover: ({x_m:.3f} m, {y_m:.3f} m) — outside ROIs")

        self.metric_label.setPixmap(
            cv_bgr_to_qpix(disp).scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

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
            self.hud_locked.setText(f"Pinned: ROI {roi_idx + 1} @ ({x_m:.3f} m, {y_m:.3f} m)")
            self._plot_time_series(roi_idx, lx, ly)
            cv2.drawMarker(disp, (ix, iy), (0, 0, 255), cv2.MARKER_CROSS, 18, 3)
        elif mode == "Vertical":
            self.profile_sel = ("V", ix)
            self.hud_locked.setText(f"Pinned: Vertical profile at x={ix * self.mpp:.3f} m")
            self._plot_vertical_profile(roi_idx, lx)
            cv2.line(disp, (ix, 0), (ix, self.overlay_h - 1), (0, 255, 0), 3)
        else:
            self.profile_sel = ("H", iy)
            self.hud_locked.setText(f"Pinned: Horizontal profile at y={iy * self.mpp:.3f} m")
            self._plot_horizontal_profile(roi_idx, ly)
            cv2.line(disp, (0, iy), (self.overlay_w - 1, iy), (0, 255, 0), 3)

        self.metric_label.setPixmap(
            cv_bgr_to_qpix(disp).scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _redraw_overlay(self):
        disp = self.overlay_bgr.copy()
        if self.profile_sel is not None:
            axis, idx = self.profile_sel
            if axis == "V":
                x = int(np.clip(idx, 0, self.overlay_w - 1))
                cv2.line(disp, (x, 0), (x, self.overlay_h - 1), (0, 255, 0), 3)
            else:
                y = int(np.clip(idx, 0, self.overlay_h - 1))
                cv2.line(disp, (0, y), (self.overlay_w - 1, y), (0, 255, 0), 3)
        if self.pin is not None:
            px, py = self.pin
            cv2.drawMarker(disp, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 18, 3)
        self.metric_label.setPixmap(
            cv_bgr_to_qpix(disp).scaled(self.metric_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def _plot_time_series(self, roi_idx, lx, ly):
        hist = self.vel_history[roi_idx]
        if not hist:
            self.plot.update_series([])
            return
        h_hist, w_hist = hist[0].shape
        if not (0 <= ly < h_hist and 0 <= lx < w_hist):
            self.plot.update_series([])
            return
        ts = np.array([f[ly, lx] for f in hist], dtype=np.float32)
        self.plot.update_series(list(ts))

    def _plot_horizontal_profile(self, roi_idx, ly):
        amap = self.avg_maps[roi_idx]
        if amap.size == 0:
            self.plot.update_hprofile([], 0)
            return
        h_amap, w_amap = amap.shape
        ly = int(np.clip(ly, 0, h_amap - 1))
        self.plot.update_hprofile(amap[ly, :], w_amap)

    def _plot_vertical_profile(self, roi_idx, lx):
        if len(self.avg_maps) < 2:
            self.plot.update_vprofile([], 0)
            return

        top = self.avg_maps[0]
        bot = self.avg_maps[1]

        if top.size == 0 or bot.size == 0:
            self.plot.update_vprofile([], 0)
            return

        h1, w1 = top.shape
        h2, w2 = bot.shape

        lx1 = int(np.clip(lx, 0, w1 - 1))
        lx2 = int(np.clip(lx, 0, w2 - 1))

        profile = np.concatenate([top[:, lx1], bot[:, lx2]], axis=0)
        self.plot.update_vprofile(profile, h1 + h2)

    def _open_validation_dialog(self):
        have_validation = (
            self.validation_depth is not None and len(self.validation_depth) > 0
            and self.measured_profile_depth is not None and len(self.measured_profile_depth) > 0
        )
        output_exists = os.path.isfile(self.output_path)

        if have_validation or output_exists:
            dlg = ValidationDialog(self.validation_path, self.output_path)
            dlg.exec_()
        else:
            QtWidgets.QMessageBox.information(
                self, "Validation",
                "No validation or output data available.\n"
                "Run the analysis first to generate velocity_output.csv.",
            )


# ===========================================================================
# Velocity aggregation
# ===========================================================================

def get_avg_velocity(velocity_sums, frame_counts, rois, mpp, increment_m=1.0):
    """
    Aggregate frame-averaged velocity maps into 1 m depth-bin averages.

    Returns a dict mapping ``depth_bin_start (m)`` to
    ``{"Velocity m/s": float}``.
    """
    height_data = {}
    current_height_m = 0.0

    for i, (rx, ry, rw, rh) in enumerate(rois):
        avg_velocity_map = velocity_sums[i] / max(1, frame_counts[i])
        roi_height_m = rh * mpp
        num_increments = int(np.floor(roi_height_m / increment_m))

        for inc in range(num_increments):
            start_y_m = current_height_m + inc * increment_m
            end_y_m = start_y_m + increment_m
            start_y_px = int((start_y_m - current_height_m) / mpp)
            end_y_px = int((end_y_m - current_height_m) / mpp)

            if start_y_px >= rh:
                break
            end_y_px = min(end_y_px, rh)

            region_velocity = avg_velocity_map[start_y_px:end_y_px, :]
            avg_vel = float(np.mean(region_velocity))
            height_key = round(start_y_m, 1)

            if height_key not in height_data:
                height_data[height_key] = {"Velocity m/s": avg_vel}
            else:
                height_data[height_key]["Velocity m/s"] = max(
                    height_data[height_key]["Velocity m/s"], avg_vel
                )

        current_height_m += roi_height_m

    return height_data


def compute_measured_vertical_profile(avg_maps, rois, mpp):
    """
    Compute the depth-averaged vertical velocity profile across all ROIs.

    Returns ``(depth_m, velocity_mps)`` — parallel 1-D arrays sorted by
    ascending depth.
    """
    rows = {}
    for (rx, ry, rw, rh), amap in zip(rois, avg_maps):
        if amap.size == 0:
            continue
        for r in range(rh):
            y_abs = ry + r
            if y_abs not in rows:
                rows[y_abs] = []
            rows[y_abs].append(float(np.nanmean(amap[r, :])))
    if not rows:
        return np.array([]), np.array([])
    y_sorted = sorted(rows.keys())
    vel_avg = np.array([np.nanmean(rows[yy]) for yy in y_sorted], dtype=float)
    depth_m = np.array(y_sorted, dtype=float) * float(mpp)
    return depth_m, vel_avg


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    args = _parse_args()

    video_path = args.video
    validation_path = args.validation
    output_path = args.output
    rotation_angle = args.rotation
    known_width_m = args.width_m

    num_regions = 2
    sfs_influences = [0.8, 0.7]
    penalty_factors = [1, 1]

    # ------------------------------------------------------------------
    # Validate video path
    # ------------------------------------------------------------------
    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        print("Usage: python main.py --video /path/to/video.MOV", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Open video and read first frame
    # ------------------------------------------------------------------
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first_frame_raw = video.read()
    if not ok or first_frame_raw is None:
        print("Error: failed to read first frame.", file=sys.stderr)
        video.release()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Spatial calibration
    # ------------------------------------------------------------------
    rotated_for_cal = rotate_image(first_frame_raw.copy(), rotation_angle)
    mpp = point_to_point(rotated_for_cal, known_metres=known_width_m)
    if mpp is None:
        print("Calibration cancelled or failed. Falling back to estimate 0.039 m/px.")
        mpp = 0.039
    else:
        print(f"Calibration result: {mpp:.8f} m/px")

    # ------------------------------------------------------------------
    # ROI selection and gamma tuning
    # ------------------------------------------------------------------
    rois = []
    gammas = []
    rotated_for_roi = rotate_image(first_frame_raw.copy(), rotation_angle)
    resized, scale = resize_to_screen(rotated_for_roi)

    for i in range(num_regions):
        roi_sel = cv2.selectROI(f"Select ROI {i + 1}", resized, showCrosshair=True)
        rx, ry, rw, rh = roi_sel
        # Store ROI in full-resolution coordinates
        rois.append((int(rx / scale), int(ry / scale), int(rw / scale), int(rh / scale)))
        gsel = select_gamma_for_roi(
            resized, (rx, ry, rw, rh),
            initial_gamma=1.0, window_name=f"Gamma ROI {i + 1}",
        )
        gammas.append(1.0 if gsel is None else float(gsel))

    cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Re-read first frame after calibration (seek back to frame 0)
    # ------------------------------------------------------------------
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = video.read()
    if not ok or first_frame is None:
        print("Error: failed to re-read first frame.", file=sys.stderr)
        video.release()
        sys.exit(1)
    first_frame = rotate_image(first_frame, rotation_angle)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = apply_clahe(prev_gray)
    prev_gray = apply_gamma_correction(prev_gray, gammas[0])

    cv2.namedWindow("Optical Flow Quiver Plot")

    velocity_sums = [None] * num_regions
    frame_counts = [0] * num_regions
    vel_history = [[] for _ in range(num_regions)]

    # ------------------------------------------------------------------
    # Main optical-flow processing loop
    # ------------------------------------------------------------------
    while True:
        ok, frame = video.read()
        if not ok or frame is None:
            break

        frame = rotate_image(frame, rotation_angle)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = apply_clahe(gray)

        for i, (rx, ry, rw, rh) in enumerate(rois):
            roi_prev = prev_gray[ry : ry + rh, rx : rx + rw]
            roi_curr = gray[ry : ry + rh, rx : rx + rw]
            roi_curr = apply_gamma_correction(roi_curr, gammas[i])

            if roi_prev.size == 0 or roi_curr.size == 0:
                continue

            flow = cv2.calcOpticalFlowFarneback(
                roi_prev, roi_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # SfS-based flow refinement
            sfs_float = roi_curr.astype(np.float32) / 255.0
            z = run_sfs_on_frame(sfs_float, iters=30)
            zx, zy = forward_gradients(z)
            downslope = np.stack([-zx, -zy], axis=-1)
            downslope /= np.linalg.norm(downslope, axis=-1, keepdims=True) + 1e-8

            flow_unit = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-8)
            alignment = np.sum(flow_unit * downslope, axis=-1)

            penalty = penalty_factors[i]
            raw_weights = np.clip((alignment + 1) / 2, 0, 1)
            raw_weights[alignment < 0] *= (1 - penalty)
            influence = sfs_influences[i]
            weights = 1.0 - influence * (1.0 - raw_weights)
            flow_refined = flow * weights[..., None]

            fx, fy = flow_refined[..., 0], flow_refined[..., 1]
            mag, _ = cv2.cartToPolar(fx, fy, angleInDegrees=True)
            velocity = mag * fps * mpp

            # Camera perspective correction
            yy_coords = np.arange(ry, ry + rh)
            row_angles_deg = np.interp(yy_coords, HEIGHT_BREAKS, HEIGHT_SCALES)
            row_scales = np.sin(np.radians(row_angles_deg))
            row_scales = np.where(row_scales < 1e-6, 1e-6, row_scales)
            velocity /= row_scales[:, None]

            if velocity_sums[i] is None:
                velocity_sums[i] = np.zeros_like(velocity, dtype=np.float32)
            velocity_sums[i] += velocity.astype(np.float32)
            frame_counts[i] += 1
            vel_history[i].append(velocity.astype(np.float32))

            avg_velocity = float(np.mean(velocity))
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ROI {i + 1}: {avg_velocity:.2f} m/s (gamma={gammas[i]:.2f})",
                (rx, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            )

            step = 12
            for yy in range(0, rh, step):
                for xx in range(0, rw, step):
                    dx = int(flow_refined[yy, xx, 0])
                    dy = int(flow_refined[yy, xx, 1])
                    if dx * dx + dy * dy > 1:
                        color = (0, 0, 255) if alignment[yy, xx] >= 0 else (0, 165, 255)
                        cv2.arrowedLine(
                            frame,
                            (rx + xx, ry + yy),
                            (rx + xx + dx, ry + yy + dy),
                            color, 1, tipLength=0.4,
                        )

        display, _ = resize_to_screen(frame.copy())
        cv2.imshow("Optical Flow Quiver Plot", display)
        prev_gray = gray.copy()

        key = cv2.waitKeyEx(10)
        if key in (27, ord("q"), ord("Q")):
            break
        try:
            if cv2.getWindowProperty("Optical Flow Quiver Plot", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    # ------------------------------------------------------------------
    # Velocity aggregation and CSV export
    # ------------------------------------------------------------------
    velocity_data = get_avg_velocity(velocity_sums, frame_counts, rois, mpp, increment_m=1.0)

    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Depth m", "Velocity m/s"])
        writer.writeheader()
        for height_key, data in sorted(velocity_data.items()):
            writer.writerow({"Depth m": height_key, "Velocity m/s": data["Velocity m/s"]})

    print(f"Velocity data exported to {output_path}")

    video.release()
    cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Build average velocity maps for the viewer
    # ------------------------------------------------------------------
    h_full, w_full = first_frame.shape[:2]
    avg_maps = []
    all_vals = []
    for i, (rx, ry, rw, rh) in enumerate(rois):
        if frame_counts[i] == 0:
            amap = np.zeros((rh, rw), dtype=np.float32)
        else:
            amap = velocity_sums[i] / float(frame_counts[i])
        avg_maps.append(amap)
        if amap.size:
            all_vals.append(amap.reshape(-1))

    global_cap = float(np.percentile(np.concatenate(all_vals), 99)) if all_vals else 1.0

    # ------------------------------------------------------------------
    # Matplotlib global style
    # ------------------------------------------------------------------
    mpl.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
    })

    # ------------------------------------------------------------------
    # Build initial overlay for the viewer
    # ------------------------------------------------------------------
    # We need a temporary ResultsViewer instance structure, so build the
    # overlay manually here using the same logic the viewer uses internally.
    cmap_table = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }
    cm_choice = cmap_table.get("JET", cv2.COLORMAP_JET)
    combined_heatmap = np.zeros((h_full, w_full, 3), dtype=np.uint8)
    combined_mask = np.zeros((h_full, w_full), dtype=bool)
    for (rx, ry, rw, rh), amap in zip(rois, avg_maps):
        if amap.size == 0:
            continue
        cap = max(1e-6, global_cap)
        capped = np.clip(amap, 0, cap)
        norm = cv2.normalize(capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cm_choice)
        combined_heatmap[ry : ry + rh, rx : rx + rw] = heat
        combined_mask[ry : ry + rh, rx : rx + rw] = True
    overlay_initial = first_frame.copy()
    overlay_initial[combined_mask] = cv2.addWeighted(
        overlay_initial[combined_mask], 0.6, combined_heatmap[combined_mask], 0.6, 0
    )

    # ------------------------------------------------------------------
    # Load validation data
    # ------------------------------------------------------------------
    ideal_depth_m, ideal_velocity = None, None
    if os.path.isfile(validation_path):
        try:
            ds, vs = [], []
            with open(validation_path, "r", newline="") as vf:
                for row in csv.DictReader(vf):
                    d = row.get("Depth m")
                    v = row.get("Velocity m/s")
                    if d is not None and v is not None:
                        ds.append(float(d))
                        vs.append(float(v))
            if ds and vs:
                ideal_depth_m = np.array(ds, dtype=float)
                ideal_velocity = np.array(vs, dtype=float)
        except Exception as exc:
            print(f"Warning: validation file read error: {exc}", file=sys.stderr)

    meas_depth_m, meas_vel_mps = compute_measured_vertical_profile(avg_maps, rois, mpp)

    # ------------------------------------------------------------------
    # Launch Qt results viewer
    # ------------------------------------------------------------------
    app = QApplication(sys.argv)
    win = ResultsViewer(
        overlay_bgr=overlay_initial,
        avg_maps=avg_maps,
        vel_history=vel_history,
        rois=rois,
        fps=fps,
        mpp=mpp,
        vmax=global_cap if global_cap > 0 else 1.0,
        cmap_name="JET",
        validation_depth=ideal_depth_m,
        validation_vel=ideal_velocity,
        measured_profile_depth=meas_depth_m,
        measured_profile_vel=meas_vel_mps,
        validation_path=validation_path,
        output_path=output_path,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
