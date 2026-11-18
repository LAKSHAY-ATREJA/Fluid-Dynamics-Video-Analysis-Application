# Changelog

## 2026-11-18

- Rejected invalid or missing recording frame rates instead of silently
  assuming 25 FPS.
- Fixed baseline quiver rendering to use the validated flow field without
  requiring experimental Shape-from-Shading state.
- Applied each ROI's selected gamma identically to both consecutive frames,
  preventing cross-ROI and temporal preprocessing mismatches.

## 2026-11-16

- Added a reusable, validated dense optical-flow core.
- Added deterministic known-translation and controlled-noise benchmarks.
- Replaced the uncalibrated synthetic demo with an encoded-video
  ground-truth evaluation and machine-readable metrics.
- Made failed spatial calibration a hard stop.
- Made Shape-from-Shading weighting and dam-specific perspective correction
  explicit experimental options instead of defaults.
- Expanded CI to run the full test suite, benchmark and headless demo.
- Added the MIT licence and an architecture decision record.


