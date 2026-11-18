"""Validated optical-flow primitives shared by the application and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FarnebackConfig:
    pyramid_scale: float = 0.5
    levels: int = 3
    window_size: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


@dataclass(frozen=True)
class QuiverSample:
    x: int
    y: int
    dx: int
    dy: int
    aligned: bool | None


@dataclass(frozen=True)
class TranslationMetrics:
    expected_dx_px: float
    expected_dy_px: float
    median_endpoint_error_px: float
    p95_endpoint_error_px: float
    median_speed_error_percent: float


def _gray_u8(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("frame must not be empty")
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        frame = cv2.cvtColor(frame, conversion)
    if frame.ndim != 2:
        raise ValueError("frame must be grayscale, BGR, or BGRA")
    if frame.dtype != np.uint8:
        raise ValueError("frame dtype must be uint8")
    return frame


def estimate_dense_flow(
    previous: np.ndarray,
    current: np.ndarray,
    config: FarnebackConfig = FarnebackConfig(),
) -> np.ndarray:
    """Estimate per-pixel displacement from previous to current."""

    previous_gray = _gray_u8(previous)
    current_gray = _gray_u8(current)
    if previous_gray.shape != current_gray.shape:
        raise ValueError("frames must have identical dimensions")
    if min(previous_gray.shape) < 16:
        raise ValueError("frames must be at least 16 pixels in each dimension")

    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        config.pyramid_scale,
        config.levels,
        config.window_size,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    )


def translation_metrics(
    flow: np.ndarray,
    *,
    expected_dx_px: float,
    expected_dy_px: float,
    border_px: int = 24,
) -> TranslationMetrics:
    """Compare a dense flow field with a known global image translation."""

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape (height, width, 2)")
    if border_px < 0 or border_px * 2 >= min(flow.shape[:2]):
        raise ValueError("border_px leaves no evaluation region")

    region = flow if border_px == 0 else flow[border_px:-border_px, border_px:-border_px]
    endpoint_error = np.hypot(
        region[..., 0] - expected_dx_px,
        region[..., 1] - expected_dy_px,
    )
    expected_speed = float(np.hypot(expected_dx_px, expected_dy_px))
    measured_speed = np.hypot(region[..., 0], region[..., 1])
    speed_error = (
        0.0
        if expected_speed == 0.0
        else abs(float(np.median(measured_speed)) - expected_speed)
        / expected_speed
        * 100.0
    )
    return TranslationMetrics(
        expected_dx_px=expected_dx_px,
        expected_dy_px=expected_dy_px,
        median_endpoint_error_px=float(np.median(endpoint_error)),
        p95_endpoint_error_px=float(np.quantile(endpoint_error, 0.95)),
        median_speed_error_percent=speed_error,
    )


def resolve_frame_rate(metadata_fps: float, *, override_fps: float | None = None) -> float:
    """Use a verified frame rate; never invent a default for physical velocities.

    The explicit override is useful for footage with broken/missing metadata,
    but must come from a trusted recording source, not a guessed value.
    """
    fps = override_fps if override_fps is not None else metadata_fps
    if fps is None or not np.isfinite(fps) or fps <= 0:
        source = "--fps" if override_fps is not None else "video metadata"
        raise ValueError(
            f"{source} frame rate must be finite and positive; "
            "supply a measured --fps value if video metadata is invalid"
        )
    return float(fps)


def sample_quiver_vectors(
    flow: np.ndarray,
    *,
    step: int = 12,
    min_magnitude_px: float = 1.0,
    alignment: np.ndarray | None = None,
) -> list[QuiverSample]:
    """Return deterministic vectors for display without assuming SfS data exists."""

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape (height, width, 2)")
    if not np.all(np.isfinite(flow)):
        raise ValueError("flow must contain only finite values")
    if step <= 0:
        raise ValueError("step must be positive")
    if not np.isfinite(min_magnitude_px) or min_magnitude_px < 0:
        raise ValueError("min_magnitude_px must be finite and non-negative")
    if alignment is not None:
        if alignment.shape != flow.shape[:2]:
            raise ValueError("alignment must match the flow height and width")
        if not np.all(np.isfinite(alignment)):
            raise ValueError("alignment must contain only finite values")

    threshold_squared = min_magnitude_px * min_magnitude_px
    samples = []
    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            dx = float(flow[y, x, 0])
            dy = float(flow[y, x, 1])
            if dx * dx + dy * dy <= threshold_squared:
                continue
            samples.append(
                QuiverSample(
                    x=x,
                    y=y,
                    dx=int(round(dx)),
                    dy=int(round(dy)),
                    aligned=None if alignment is None else bool(alignment[y, x] >= 0),
                )
            )
    return samples


def velocity_magnitude(
    flow: np.ndarray,
    *,
    frames_per_second: float,
    metres_per_pixel: float,
) -> np.ndarray:
    """Convert displacement in px/frame to planar image speed in m/s."""

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape (height, width, 2)")
    if frames_per_second <= 0 or not np.isfinite(frames_per_second):
        raise ValueError("frames_per_second must be finite and positive")
    if metres_per_pixel <= 0 or not np.isfinite(metres_per_pixel):
        raise ValueError("metres_per_pixel must be finite and positive")
    return (
        np.hypot(flow[..., 0], flow[..., 1])
        * frames_per_second
        * metres_per_pixel
    )

