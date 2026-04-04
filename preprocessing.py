# preprocessing.py
"""
Image preprocessing utilities used by main.py and SFSImplementation.py.

Functions cover frame rotation, screen-fitting resize, CLAHE contrast
enhancement, gamma correction, interactive spatial calibration, and an
interactive gamma-selection widget.
"""

import cv2
import numpy as np
import math


def rotate_image(img, angle):
    """Rotate *img* by *angle* degrees (0 / 90 / 180 / 270)."""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img.copy()


def resize_to_screen(img, screen_res=(1280, 720)):
    """
    Return a copy of *img* scaled to fit within *screen_res* while preserving
    aspect ratio, together with the uniform scale factor applied.
    """
    if img is None or img.size == 0:
        raise ValueError("resize_to_screen received an empty image")
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    new_size = (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
    return cv2.resize(img, new_size), scale


def apply_clahe(img):
    """Apply CLAHE contrast-limited adaptive histogram equalization to *img*."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def apply_gamma_correction(img, gamma=1.3):
    """
    Apply a domain-specific gamma correction to *img*.

    The transformation uses exponent ``3 / gamma`` which, for the typical
    parameter ranges used in this application (0.55--5.0), provides:

    - ``gamma`` > 1  (e.g. 4.5 for clear water): moderate brightening that
      enhances sub-surface flow patterns without saturating highlights.
    - ``gamma`` < 1  (e.g. 0.6 for turbulent white water): aggressive
      darkening that suppresses specular reflections.

    Note: this is intentionally not the standard ``1/gamma`` photographic
    convention.  Both the ROI selection widget and the SfS module rely on
    this behaviour.
    """
    gamma = max(1e-6, gamma)
    inv_gamma = 3.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.float32
    ).clip(0, 255).astype("uint8")
    return cv2.LUT(img, table)


def point_to_point(frame0, known_metres=12.25, win="Calibrate", rotate=0):
    """
    Interactive two-click spatial calibration.

    Displays *frame0* in a resizable window and asks the user to click two
    points separated by *known_metres*.  Returns metres-per-pixel (float), or
    ``None`` if the user cancels (ESC / Q) or the two points coincide.

    Parameters
    ----------
    frame0 : ndarray
        The reference frame (BGR, already rotated if necessary).
    known_metres : float
        Real-world distance in metres between the two clicked points.
    win : str
        OpenCV window name.
    rotate : int
        Additional rotation to apply before display (0 / 90 / 180 / 270).
    """
    if frame0 is None or frame0.size == 0:
        return None

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    img = rotate_image(frame0.copy(), rotate)

    pts = []

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((mx, my))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        d = img.copy()
        for p in pts:
            cv2.circle(d, p, 8, (0, 255, 0), -1, cv2.LINE_AA)
        if len(pts) == 2:
            cv2.line(d, pts[0], pts[1], (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            d,
            "Click both ends of the known distance. ENTER=accept  ESC/Q=cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(win, d)
        k = cv2.waitKey(30) & 0xFF

        if k in (27, ord("q")):  # ESC or Q: cancel
            pts.clear()
            cv2.setMouseCallback(win, lambda *a: None)
            cv2.destroyWindow(win)
            return None

        if k in (13, 32) and len(pts) == 2:  # ENTER or SPACE with two points
            dx = pts[0][0] - pts[1][0]
            dy = pts[0][1] - pts[1][1]
            pix = math.hypot(dx, dy)
            cv2.setMouseCallback(win, lambda *a: None)
            cv2.destroyWindow(win)
            if pix > 0:
                return known_metres / pix
            return None  # coincident points


def select_gamma_for_roi(
    resized_img,
    roi_resized,
    initial_gamma=1.0,
    window_name="Gamma Selector",
    max_display=600,
):
    """
    Interactive gamma-adjustment widget for a single region of interest.

    Shows the original CLAHE-enhanced crop alongside a live gamma-corrected
    preview.  A trackbar maps positions 0--2000 to gamma values 0.00--20.00
    in steps of 0.01.

    Parameters
    ----------
    resized_img : ndarray
        The full resized BGR frame.
    roi_resized : tuple
        ``(x, y, w, h)`` of the ROI in *resized_img* coordinates.
    initial_gamma : float
        Starting gamma value shown on the trackbar.
    window_name : str
        OpenCV window name.
    max_display : int
        Maximum pixel size of the longest ROI dimension in the preview.

    Returns
    -------
    float or None
        Accepted gamma, or ``None`` if the user cancels.
    """
    x, y, w, h = roi_resized
    if w <= 0 or h <= 0:
        return None

    crop = resized_img[y : y + h, x : x + w].copy()
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        clahe_gray = apply_clahe(gray)
    except Exception:
        clahe_gray = gray

    largest_dim = max(w, h)
    scale_float = max(1.0, float(max_display) / float(largest_dim))
    scale_float = min(scale_float, 8.0)
    disp_w = max(1, int(round(w * scale_float)))
    disp_h = max(1, int(round(h * scale_float)))

    pos_max = 2000
    init_pos = int(round(max(0.0, min(20.0, initial_gamma)) * 100))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w * 2 + 20, disp_h + 80)
    cv2.createTrackbar("Gamma", window_name, init_pos, pos_max, lambda v: None)

    last_pos = None
    accepted_gamma = None

    def make_display(gamma_val):
        gamma_safe = max(0.01, gamma_val)
        try:
            preview = apply_gamma_correction(clahe_gray, gamma=gamma_safe)
        except Exception:
            lut = np.array(
                [((i / 255.0) ** (3.0 / gamma_safe)) * 255 for i in range(256)],
                dtype=np.float32,
            ).clip(0, 255).astype(np.uint8)
            preview = cv2.LUT(clahe_gray, lut)

        orig_up = cv2.resize(clahe_gray, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        prev_up = cv2.resize(preview, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        left = cv2.cvtColor(orig_up, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(prev_up, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            left, "Original (CLAHE)", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            right, f"Preview (gamma={gamma_safe:.2f})", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        combined = np.hstack([left, right])
        overlay = np.zeros((combined.shape[0] + 60, combined.shape[1], 3), dtype=np.uint8)
        overlay[: combined.shape[0], :, :] = combined
        instr = "ENTER/SPACE accept  |  ESC/Q cancel  |  Trackbar: 0.00 -> 20.00 (step 0.01)"
        cv2.putText(
            overlay, instr, (8, combined.shape[0] + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return overlay

    while True:
        pos = cv2.getTrackbarPos("Gamma", window_name)
        pos = max(0, min(pos_max, pos))

        if pos != last_pos:
            gamma = max(0.01, pos / 100.0)
            disp = make_display(gamma)
            cv2.imshow(window_name, disp)
            last_pos = pos

        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32):
            accepted_gamma = max(0.01, pos / 100.0)
            break
        if k in (27, ord("q")):
            accepted_gamma = None
            break

    cv2.destroyWindow(window_name)
    return accepted_gamma
