import numpy as np
import pytest

from preprocessing import apply_gamma_correction, resize_to_screen, rotate_image


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


def test_gamma_correction_preserves_shape_and_dtype():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    corrected = apply_gamma_correction(image, gamma=1.3)
    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint8
