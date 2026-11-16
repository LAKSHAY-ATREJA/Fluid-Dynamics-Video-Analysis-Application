# ADR 0001: Validated baseline before research corrections

- **Status:** Accepted
- **Date:** 16 November 2026

## Context

The original application combined Farneback optical flow, Shape-from-Shading
direction weighting and a row-angle correction derived from one dam geometry.
Its public demo regenerated random texture every frame and printed a physical
velocity without a known-motion reference. That output could not establish
measurement accuracy.

## Decision

1. Use raw dense optical flow as the default observable baseline.
2. Require an explicit spatial calibration. Failed interactive calibration
   stops the run instead of substituting a hard-coded scale.
3. Keep Shape-from-Shading weighting behind
   `--experimental-sfs-refinement`.
4. Keep the dam row-angle table behind
   `--dam-perspective-correction`.
5. Validate the default path with deterministic global translations, controlled
   noise and a video encode/decode round trip.
6. Report endpoint and speed error against generated ground truth.

## Consequences

The default application makes a narrower but testable claim: it estimates
planar image motion and converts that motion using a user-supplied calibration.
It does not claim publicly validated water velocity or physical depth.

The optional research corrections remain available for future evaluation
against suitable field data. They must not be described as improving accuracy
until a reproducible comparison demonstrates that improvement.

