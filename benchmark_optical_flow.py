"""Deterministic known-motion benchmark for the optical-flow component."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from flow_analysis import estimate_dense_flow, translation_metrics


@dataclass(frozen=True)
class Case:
    name: str
    dx_px: float
    dy_px: float
    noise_sigma: float
    max_p95_error_px: float


CASES = (
    Case("subpixel", 0.5, 0.25, 0.0, 0.08),
    Case("diagonal", 2.0, 1.0, 0.0, 0.05),
    Case("larger-motion", 8.0, 0.0, 0.0, 0.10),
    Case("sensor-noise", 2.0, 1.0, 12.0, 0.55),
)


def _texture(seed: int, height: int = 192, width: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(127.0, 35.0, (height, width)).clip(0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (0, 0), 1.1)


def _translated(image: np.ndarray, case: Case, seed: int) -> np.ndarray:
    transform = np.float32([[1.0, 0.0, case.dx_px], [0.0, 1.0, case.dy_px]])
    moved = cv2.warpAffine(
        image,
        transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if case.noise_sigma:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, case.noise_sigma, moved.shape)
        moved = np.clip(moved.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return moved


def run_benchmark() -> dict[str, object]:
    base = _texture(seed=20261116)
    results = []
    for index, case in enumerate(CASES):
        current = _translated(base, case, seed=20261117 + index)
        metrics = translation_metrics(
            estimate_dense_flow(base, current),
            expected_dx_px=case.dx_px,
            expected_dy_px=case.dy_px,
        )
        row = asdict(metrics)
        row.update(
            name=case.name,
            noise_sigma=case.noise_sigma,
            max_p95_error_px=case.max_p95_error_px,
            passed=metrics.p95_endpoint_error_px <= case.max_p95_error_px,
        )
        results.append(row)
    return {
        "benchmark": "deterministic global-translation optical flow",
        "seed": 20261116,
        "opencv_version": cv2.__version__,
        "limitations": [
            "synthetic texture is not turbulent water",
            "camera calibration and perspective correction are not evaluated",
            "shape-from-shading refinement is not evaluated",
        ],
        "results": results,
        "passed": all(row["passed"] for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if a threshold fails"
    )
    args = parser.parse_args()
    report = run_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(args.check and not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
