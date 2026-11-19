import numpy as np
import pytest

from velocity_aggregation import (
    compute_measured_vertical_profile,
    get_avg_velocity,
)


def test_depth_bins_use_absolute_roi_geometry_and_keep_partial_bins():
    velocity_sums = [
        np.full((3, 2), 2.0),
        np.full((2, 2), 4.0),
    ]
    rois = [(0, 2, 2, 3), (0, 8, 2, 2)]

    result = get_avg_velocity(
        velocity_sums,
        [1, 1],
        rois,
        mpp=0.4,
        increment_m=1.0,
    )

    assert result == {
        0.0: {"Velocity m/s": pytest.approx(2.0)},
        1.0: {"Velocity m/s": pytest.approx(2.0)},
        3.0: {"Velocity m/s": pytest.approx(4.0)},
    }


def test_overlapping_rois_are_weighted_by_observed_pixels_not_maximum():
    result = get_avg_velocity(
        [np.full((1, 2), 2.0), np.full((1, 1), 8.0)],
        [1, 1],
        [(0, 5, 2, 1), (4, 5, 1, 1)],
        mpp=0.1,
        increment_m=1.0,
    )
    assert result[0.0]["Velocity m/s"] == pytest.approx(4.0)


def test_unobserved_roi_is_omitted_instead_of_exported_as_zero():
    result = get_avg_velocity(
        [None, np.full((1, 2), 3.0)],
        [0, 2],
        [(0, 0, 2, 1), (0, 10, 2, 1)],
        mpp=0.1,
    )
    assert result == {1.0: {"Velocity m/s": pytest.approx(1.5)}}


def test_vertical_profile_uses_absolute_rows_and_pixel_weighted_overlap():
    depth, velocity = compute_measured_vertical_profile(
        [np.full((2, 2), 2.0), np.full((1, 1), 8.0)],
        [(0, 5, 2, 2), (4, 5, 1, 1)],
        mpp=0.1,
    )
    assert depth == pytest.approx([0.5, 0.6])
    assert velocity == pytest.approx([4.0, 2.0])


def test_aggregation_rejects_shape_and_scale_errors():
    with pytest.raises(ValueError, match="shape must match"):
        get_avg_velocity(
            [np.ones((2, 2))],
            [1],
            [(0, 0, 3, 2)],
            mpp=0.1,
        )
    with pytest.raises(ValueError, match="mpp must be"):
        compute_measured_vertical_profile(
            [np.ones((1, 1))],
            [(0, 0, 1, 1)],
            mpp=float("nan"),
        )
