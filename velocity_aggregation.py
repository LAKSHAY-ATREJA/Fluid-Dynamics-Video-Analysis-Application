"""Geometry-aware aggregation of calibrated velocity maps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def _validate_scale(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_roi(roi, *, index: int) -> tuple[int, int, int, int]:
    if len(roi) != 4:
        raise ValueError(f"ROI {index} must contain x, y, width and height")
    x, y, width, height = (int(value) for value in roi)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"ROI {index} must be a positive rectangle")
    return x, y, width, height


def _frame_average(
    velocity_sum: np.ndarray | None,
    frame_count: int,
    roi: tuple[int, int, int, int],
    *,
    index: int,
) -> np.ndarray | None:
    _x, _y, width, height = roi
    if frame_count < 0:
        raise ValueError(f"frame count {index} must not be negative")
    if frame_count == 0:
        if velocity_sum is not None and np.asarray(velocity_sum).size:
            raise ValueError(
                f"velocity sum {index} contains data but its frame count is zero"
            )
        return None
    if velocity_sum is None:
        raise ValueError(f"velocity sum {index} is missing for processed frames")

    values = np.asarray(velocity_sum, dtype=np.float64)
    if values.shape != (height, width):
        raise ValueError(
            f"velocity sum {index} shape must match ROI height and width"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"velocity sum {index} must contain only finite values")
    return values / float(frame_count)


def get_avg_velocity(
    velocity_sums: Sequence[np.ndarray | None],
    frame_counts: Sequence[int],
    rois: Sequence[tuple[int, int, int, int]],
    mpp: float,
    increment_m: float = 1.0,
) -> dict[float, dict[str, float]]:
    """Aggregate maps into physical-depth bins using absolute image rows.

    Every finite pixel has equal weight. Overlapping ROIs therefore contribute
    their observed pixels rather than selecting whichever ROI has the largest
    mean. Partial bins are retained and bins with no processed frames are
    omitted.
    """

    if not (len(velocity_sums) == len(frame_counts) == len(rois)):
        raise ValueError("velocity sums, frame counts and ROIs must have equal length")
    metres_per_pixel = _validate_scale(mpp, "mpp")
    bin_width = _validate_scale(increment_m, "increment_m")

    totals = defaultdict(float)
    sample_counts = defaultdict(int)
    for index, (velocity_sum, frame_count, raw_roi) in enumerate(
        zip(velocity_sums, frame_counts, rois)
    ):
        roi = _validate_roi(raw_roi, index=index)
        average = _frame_average(
            velocity_sum, int(frame_count), roi, index=index
        )
        if average is None:
            continue
        _x, y, _width, height = roi
        for local_y in range(height):
            depth_m = (y + local_y) * metres_per_pixel
            bin_index = int(np.floor(depth_m / bin_width))
            row = average[local_y, :]
            totals[bin_index] += float(np.sum(row, dtype=np.float64))
            sample_counts[bin_index] += int(row.size)

    output = {}
    for bin_index in sorted(totals):
        depth_start = float(bin_index * bin_width)
        output[depth_start] = {
            "Velocity m/s": totals[bin_index] / sample_counts[bin_index]
        }
    return output


def compute_measured_vertical_profile(
    avg_maps: Sequence[np.ndarray],
    rois: Sequence[tuple[int, int, int, int]],
    mpp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-image-row velocity means at calibrated depth coordinates."""

    if len(avg_maps) != len(rois):
        raise ValueError("average maps and ROIs must have equal length")
    metres_per_pixel = _validate_scale(mpp, "mpp")

    totals = defaultdict(float)
    sample_counts = defaultdict(int)
    for index, (raw_map, raw_roi) in enumerate(zip(avg_maps, rois)):
        roi = _validate_roi(raw_roi, index=index)
        values = np.asarray(raw_map, dtype=np.float64)
        if values.size == 0:
            continue
        _x, y, width, height = roi
        if values.shape != (height, width):
            raise ValueError(
                f"average map {index} shape must match ROI height and width"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"average map {index} must contain only finite values")
        for local_y in range(height):
            absolute_y = y + local_y
            row = values[local_y, :]
            totals[absolute_y] += float(np.sum(row, dtype=np.float64))
            sample_counts[absolute_y] += int(row.size)

    if not totals:
        return np.array([], dtype=float), np.array([], dtype=float)
    rows = np.array(sorted(totals), dtype=float)
    velocities = np.array(
        [totals[int(row)] / sample_counts[int(row)] for row in rows],
        dtype=float,
    )
    return rows * metres_per_pixel, velocities
