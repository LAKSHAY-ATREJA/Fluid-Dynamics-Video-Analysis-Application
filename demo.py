"""Credential-free optical-flow demo with measurable ground truth.

The generated video translates one deterministic textured image by a known
subpixel displacement per frame. The demo reports endpoint and speed error;
it does not model turbulent water or validate physical depth.
"""

from __future__ import annotations

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from flow_analysis import estimate_dense_flow, translation_metrics, velocity_magnitude

VIDEO_PATH = "demo_input.mp4"
OUT_HEATMAP = "demo_velocity_heatmap.png"
OUT_PROFILE = "demo_velocity_profile.png"
OUT_REPORT = "demo_metrics.json"

FRAME_W = 320
FRAME_H = 240
NUM_FRAMES = 40
FPS = 25.0
METRES_PER_PIXEL = 0.01
DX_PER_FRAME = 0.75
DY_PER_FRAME = 0.15
BORDER_PX = 32
MAX_P95_ENDPOINT_ERROR_PX = 0.8


def _base_texture() -> np.ndarray:
    rng = np.random.default_rng(20261116)
    texture = (
        rng.normal(127.0, 35.0, (FRAME_H, FRAME_W))
        .clip(0, 255)
        .astype(np.uint8)
    )
    return cv2.GaussianBlur(texture, (0, 0), 1.1)


def generate_video(path: str) -> None:
    base = _base_texture()
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (FRAME_W, FRAME_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create demo video: {path}")
    for frame_index in range(NUM_FRAMES):
        transform = np.float32(
            [
                [1.0, 0.0, frame_index * DX_PER_FRAME],
                [0.0, 1.0, frame_index * DY_PER_FRAME],
            ]
        )
        gray = cv2.warpAffine(
            base,
            transform,
            (FRAME_W, FRAME_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()


def run_demo() -> dict[str, object]:
    generate_video(VIDEO_PATH)
    capture = cv2.VideoCapture(VIDEO_PATH)
    ok, previous = capture.read()
    if not ok or previous is None:
        capture.release()
        raise RuntimeError("could not read the generated demo video")

    velocity_sum = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
    per_frame = []
    while True:
        ok, current = capture.read()
        if not ok or current is None:
            break
        flow = estimate_dense_flow(previous, current)
        per_frame.append(
            translation_metrics(
                flow,
                expected_dx_px=DX_PER_FRAME,
                expected_dy_px=DY_PER_FRAME,
                border_px=BORDER_PX,
            )
        )
        velocity_sum += velocity_magnitude(
            flow,
            frames_per_second=FPS,
            metres_per_pixel=METRES_PER_PIXEL,
        )
        previous = current
    capture.release()

    if len(per_frame) != NUM_FRAMES - 1:
        raise RuntimeError(
            f"expected {NUM_FRAMES - 1} frame pairs, processed {len(per_frame)}"
        )

    average_velocity = (velocity_sum / len(per_frame)).astype(np.float32)
    median_endpoint = float(
        np.median([metric.median_endpoint_error_px for metric in per_frame])
    )
    p95_endpoint = float(
        np.quantile([metric.p95_endpoint_error_px for metric in per_frame], 0.95)
    )
    median_speed_error = float(
        np.median([metric.median_speed_error_percent for metric in per_frame])
    )
    expected_speed = (
        float(np.hypot(DX_PER_FRAME, DY_PER_FRAME)) * FPS * METRES_PER_PIXEL
    )

    interior = average_velocity[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX]
    vmax = float(np.quantile(interior, 0.99))
    normed = cv2.normalize(
        np.clip(average_velocity, 0, vmax), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    cv2.imwrite(OUT_HEATMAP, cv2.applyColorMap(normed, cv2.COLORMAP_VIRIDIS))

    profile = average_velocity[FRAME_H // 2, BORDER_PX:-BORDER_PX]
    positions = np.arange(profile.size) * METRES_PER_PIXEL
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(positions, profile, label="measured")
    axis.axhline(expected_speed, color="black", linestyle="--", label="expected")
    axis.set(
        xlabel="image-plane position (m)",
        ylabel="planar speed (m/s)",
        title="Known-translation optical-flow demo",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT_PROFILE, dpi=150)
    plt.close(fig)

    report = {
        "input": {
            "frames": NUM_FRAMES,
            "fps": FPS,
            "metres_per_pixel": METRES_PER_PIXEL,
            "translation_px_per_frame": [DX_PER_FRAME, DY_PER_FRAME],
            "expected_planar_speed_mps": expected_speed,
        },
        "observed": {
            "median_endpoint_error_px": median_endpoint,
            "p95_endpoint_error_px": p95_endpoint,
            "median_speed_error_percent": median_speed_error,
        },
        "thresholds": {
            "max_p95_endpoint_error_px": MAX_P95_ENDPOINT_ERROR_PX,
        },
        "limitations": [
            "synthetic translation is not turbulent water",
            "the spatial scale is declared test data, not a measured scene calibration",
            "shape-from-shading and dam-specific perspective correction are excluded",
        ],
        "passed": p95_endpoint <= MAX_P95_ENDPOINT_ERROR_PX,
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run_demo()["passed"] else 1)

