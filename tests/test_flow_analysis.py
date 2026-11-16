import numpy as np
import pytest

from benchmark_optical_flow import run_benchmark
from flow_analysis import estimate_dense_flow, translation_metrics, velocity_magnitude


def test_known_translation_benchmark_meets_declared_thresholds():
    report = run_benchmark()
    assert report["passed"] is True
    assert {row["name"] for row in report["results"]} == {
        "subpixel",
        "diagonal",
        "larger-motion",
        "sensor-noise",
    }


def test_velocity_conversion_uses_frame_rate_and_spatial_scale():
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    flow[..., 0] = 3.0
    flow[..., 1] = 4.0
    speed = velocity_magnitude(
        flow, frames_per_second=25.0, metres_per_pixel=0.04
    )
    assert speed == pytest.approx(np.full((2, 2), 5.0))


def test_invalid_frames_and_calibration_are_rejected():
    frame = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="identical"):
        estimate_dense_flow(frame, np.zeros((33, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="positive"):
        velocity_magnitude(
            np.zeros((2, 2, 2)),
            frames_per_second=0,
            metres_per_pixel=0.04,
        )
    with pytest.raises(ValueError, match="evaluation region"):
        translation_metrics(
            np.zeros((32, 32, 2)),
            expected_dx_px=1,
            expected_dy_px=0,
            border_px=16,
        )
