import numpy as np
import pytest

from benchmark_optical_flow import run_benchmark
from flow_analysis import (
    estimate_dense_flow,
    resolve_frame_rate,
    sample_quiver_vectors,
    translation_metrics,
    velocity_magnitude,
)


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


def test_frame_rate_uses_valid_metadata_or_explicit_measured_override():
    assert resolve_frame_rate(29.97) == pytest.approx(29.97)
    assert resolve_frame_rate(0.0, override_fps=24.0) == pytest.approx(24.0)
    assert resolve_frame_rate(float("nan"), override_fps=59.94) == pytest.approx(59.94)


@pytest.mark.parametrize("invalid", [0.0, -25.0, float("nan"), float("inf"), -float("inf")])
def test_missing_or_invalid_video_fps_is_never_silently_guessed(invalid):
    with pytest.raises(ValueError, match="video metadata frame rate"):
        resolve_frame_rate(invalid)


@pytest.mark.parametrize("invalid", [0.0, -24.0, float("nan"), float("inf")])
def test_invalid_explicit_fps_override_is_rejected_even_with_good_metadata(invalid):
    with pytest.raises(ValueError, match="--fps frame rate"):
        resolve_frame_rate(25.0, override_fps=invalid)



def test_quiver_sampling_works_without_experimental_alignment():
    flow = np.zeros((24, 24, 2), dtype=np.float32)
    flow[0, 0] = [2.4, -1.6]
    samples = sample_quiver_vectors(flow, step=12)
    assert len(samples) == 1
    sample = samples[0]
    assert (sample.x, sample.y, sample.dx, sample.dy) == (0, 0, 2, -2)
    assert sample.aligned is None


def test_quiver_sampling_preserves_optional_alignment_and_validates_shape():
    flow = np.zeros((24, 24, 2), dtype=np.float32)
    flow[12, 12] = [3.0, 0.0]
    alignment = np.ones((24, 24), dtype=np.float32)
    alignment[12, 12] = -0.25

    samples = sample_quiver_vectors(flow, step=12, alignment=alignment)
    assert len(samples) == 1
    assert samples[0].aligned is False

    with pytest.raises(ValueError, match="match the flow"):
        sample_quiver_vectors(flow, alignment=np.zeros((12, 12)))

