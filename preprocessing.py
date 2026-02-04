# preprocessing.py
import cv2
import numpy as np
import math

def rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def resize_to_screen(img, screen_res=(1280, 720)):
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, new_size), scale

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def apply_gamma_correction(img, gamma=1.3):
    inv_gamma = 3 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def point_to_point(frame0, known_metres=12.25, win="Calibrate", rotate=0):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    img = rotate_image(frame0.copy(), rotate)
    fallback = None

    pts = []
    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((mx, my))

    cv2.setMouseCallback(win, on_mouse)
    while True:
        d = img.copy()

        for p in pts:
            cv2.circle(d, p, 8, (0,255,0), -1, cv2.LINE_AA)

        if len(pts) == 2:
            cv2.line(d, pts[0], pts[1], (0,255,0), 2, cv2.LINE_AA)

        cv2.putText(d, "Click both ends. ENTER=accept  ESC=cancel", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow(win, d)
        k = cv2.waitKey(30) & 0xFF

        if k in (27, ord('q')):  # cancel two-click
            pts = []
            cv2.setMouseCallback(win, lambda *args: None)
            break

        if k in (13, 32) and len(pts) == 2:  # ENTER or SPACE with two points
            dx = pts[0][0] - pts[1][0]
            dy = pts[0][1] - pts[1][1]
            pix = math.hypot(dx, dy)
            cv2.setMouseCallback(win, lambda *args: None)
            cv2.destroyWindow(win)
            if pix > 0:
                return known_metres / pix
            else:
                return fallback
                    
def select_gamma_for_roi(resized_img, roi_resized, initial_gamma=1.0,
                         window_name="Gamma Selector", max_display=600):
    x, y, w, h = roi_resized
    if w <= 0 or h <= 0:
        return None

    crop = resized_img[y:y+h, x:x+w].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    try:
        clahe_gray = apply_clahe(gray)
    except Exception:
        clahe_gray = gray

    largest_dim = max(w, h)
    scale_float = max(1.0, float(max_display) / float(largest_dim))
    scale_float = min(scale_float, 8.0)
    disp_w = int(round(w * scale_float))
    disp_h = int(round(h * scale_float))

    pos_min = 0
    pos_max = 2000
    init_pos = int(round(max(0.0, min(20.0, initial_gamma)) * 100))

    win = window_name
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w * 2 + 20, disp_h + 80)
    cv2.createTrackbar("Gamma", win, init_pos, pos_max, lambda v: None)

    last_pos = None
    accepted_gamma = None

    def make_display(gamma_val):
        gamma_safe = max(0.01, gamma_val)

        try:
            preview = apply_gamma_correction(clahe_gray, gamma=gamma_safe)
        except Exception:
            lut = np.array([((i / 255.0) ** (1.0 / gamma_safe)) * 255 for i in range(256)]).astype(np.uint8)
            preview = cv2.LUT(clahe_gray, lut)

        orig_up = cv2.resize(clahe_gray, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        prev_up = cv2.resize(preview, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        left = cv2.cvtColor(orig_up, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(prev_up, cv2.COLOR_GRAY2BGR)

        cv2.putText(left, "Original (CLAHE)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(right, f"Preview (gamma={gamma_safe:.2f})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        combined = np.hstack([left, right])
        overlay = np.zeros((combined.shape[0] + 60, combined.shape[1], 3), dtype=np.uint8)
        overlay[0:combined.shape[0], :, :] = combined
        instr = "ENTER/SPACE accept  |  ESC/Q cancel  |  Trackbar: 0.00 -> 20.00 (step 0.01)"
        cv2.putText(overlay, instr, (8, combined.shape[0] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
        return overlay

    while True:
        pos = cv2.getTrackbarPos("Gamma", win)
        if pos < pos_min:
            pos = pos_min
            cv2.setTrackbarPos("Gamma", win, pos)
        if pos > pos_max:
            pos = pos_max
            cv2.setTrackbarPos("Gamma", win, pos)

        if pos != last_pos:
            gamma = pos / 100.0
            gamma_for_preview = gamma if gamma >= 0.01 else 0.01
            disp = make_display(gamma_for_preview)
            cv2.imshow(win, disp)
            last_pos = pos

        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32):
            accepted_gamma = max(0.01, pos / 100.0)
            break
        if k in (27, ord('q')):
            accepted_gamma = None
            break

    cv2.destroyWindow(win)
    return accepted_gamma