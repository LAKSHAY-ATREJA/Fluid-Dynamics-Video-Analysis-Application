"""Offscreen regression tests for the Qt/Matplotlib results viewer."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from main import LivePlot


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _visible_text(axes):
    for axis in axes:
        yield axis.title
        yield axis.xaxis.label
        yield axis.yaxis.label
        xmin, xmax = sorted(axis.get_xlim())
        ymin, ymax = sorted(axis.get_ylim())
        yield from (
            label for location, label in zip(axis.get_xticks(), axis.get_xticklabels())
            if xmin <= location <= xmax
        )
        yield from (
            label for location, label in zip(axis.get_yticks(), axis.get_yticklabels())
            if ymin <= location <= ymax
        )
        legend = axis.get_legend()
        if legend is not None:
            yield from legend.get_texts()


def test_live_plot_labels_fit_default_results_panel(qt_app):
    """Labels must remain visible at the panel size used by ResultsViewer."""
    plot = LivePlot(fps=25.0, mpp=0.01)
    plot.resize(353, 882)
    plot.show()
    qt_app.processEvents()

    values = np.linspace(0.5, 1.3, 100)
    plot.update_hprofile(values, pixel_len=len(values))
    plot.draw()
    qt_app.processEvents()

    renderer = plot.figure.canvas.get_renderer()
    # Text extents and renderer dimensions share physical-pixel coordinates;
    # Qt may expose the canvas size in logical pixels on HiDPI systems.
    width, height = renderer.width, renderer.height
    tolerance = 1.0

    for label in _visible_text(plot.figure.axes):
        if not label.get_visible() or not label.get_text():
            continue
        bounds = label.get_window_extent(renderer=renderer)
        assert bounds.x0 >= -tolerance, f"{label.get_text()!r} clipped on left"
        assert bounds.y0 >= -tolerance, f"{label.get_text()!r} clipped on bottom"
        assert bounds.x1 <= width + tolerance, f"{label.get_text()!r} clipped on right"
        assert bounds.y1 <= height + tolerance, f"{label.get_text()!r} clipped on top"

    plot.close()
