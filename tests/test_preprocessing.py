import numpy as np
import pytest

from preprocessing import (
    apply_gamma_correction,
    map_roi_to_frame,
    prepare_flow_pair,
    resize_to_screen,
    rotate_image,
    validate_selected_gamma,
)


def test_rotate_image_90_clockwise():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    rotated = rotate_image(image, 90)
    assert rotated.tolist() == [[3, 1], [4, 2]]


def test_resize_to_screen_preserves_aspect_ratio():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized, scale = resize_to_screen(image, screen_res=(100, 100))
    assert resized.shape[:2] == (50, 100)
    assert scale == pytest.approx(0.5)


def test_resize_to_screen_rejects_empty_image():
    with pytest.raises(ValueError):
        resize_to_screen(np.array([], dtype=np.uint8))


def test_map_roi_to_frame_preserves_selected_extent_when_scaled():
    assert map_roi_to_frame((10, 20, 31, 40), 0.5, (200, 300, 3)) == (
        20, 40, 62, 80
    )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ((0, 0, 0, 0), "cancelled or empty"),
        ((-1, 0, 5, 5), "cancelled or empty"),
        ((95, 10, 10, 10), "inside the source frame"),
        ((0, 0, np.nan, 5), "finite"),
    ],
)
def test_map_roi_to_frame_rejects_invalid_selections(selection, message):
    with pytest.raises(ValueError, match=message):
        map_roi_to_frame(selection, 1.0, (100, 100, 3))


def test_validate_selected_gamma_rejects_cancel_and_invalid_values():
    with pytest.raises(ValueError, match="cancelled"):
        validate_selected_gamma(None)
    with pytest.raises(ValueError, match="finite and positive"):
        validate_selected_gamma(float("inf"))
    with pytest.raises(ValueError, match="finite and positive"):
        validate_selected_gamma(0)
    assert validate_selected_gamma(1.25) == pytest.approx(1.25)


def test_gamma_correction_preserves_shape_and_dtype():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    corrected = apply_gamma_correction(image, gamma=1.3)
    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint8



def test_prepare_flow_pair_applies_the_same_roi_gamma_to_both_frames():
    previous = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    current = np.flipud(previous).copy()
    roi = (2, 1, 4, 5)

    previous_roi, current_roi = prepare_flow_pair(
        previous, current, roi, gamma=0.75
    )

    expected_previous = apply_gamma_correction(previous[1:6, 2:6], gamma=0.75)
    expected_current = apply_gamma_correction(current[1:6, 2:6], gamma=0.75)
    assert np.array_equal(previous_roi, expected_previous)
    assert np.array_equal(current_roi, expected_current)


def test_prepare_flow_pair_keeps_roi_specific_gamma_isolated():
    frame = np.arange(256, dtype=np.uint8).reshape(16, 16)
    low_gamma, _ = prepare_flow_pair(frame, frame, (0, 0, 8, 8), gamma=0.6)
    high_gamma, _ = prepare_flow_pair(frame, frame, (0, 0, 8, 8), gamma=4.5)
    assert not np.array_equal(low_gamma, high_gamma)


@pytest.mark.parametrize(
    "roi",
    [(-1, 0, 4, 4), (0, 0, 0, 4), (0, 0, 4, 0), (14, 14, 4, 4)],
)
def test_prepare_flow_pair_rejects_invalid_rois(roi):
    frame = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="inside both frames"):
        prepare_flow_pair(frame, frame, roi, gamma=1.0)

