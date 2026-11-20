# Changelog

## 2026-11-20

- Prevented the results viewer's profile and histogram labels from being
  clipped when the plot panel is narrower than Matplotlib's initial size.
- Added an offscreen Qt render regression test for the default results-panel
  dimensions and populated profile data.
- Rejected cancelled, empty, non-finite and out-of-frame ROI selections before
  optical-flow processing, and made gamma-selection cancellation stop cleanly.

## 2026-11-19

- Replaced sequential ROI stacking with depth bins derived from actual
  full-frame row coordinates.
- Retained partial bins, omitted unobserved ROIs, and combined overlapping
  regions with per-pixel weighting instead of selecting the maximum mean.
- Added headless tests for aggregation geometry, overlaps, gaps and invalid
  inputs.

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


