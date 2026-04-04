# Fluid Dynamics Video Analysis Application

A desktop application for measuring water flow velocities from video footage.
The system combines dense optical flow (Farneback method) with Shape-from-Shading
(SfS) depth estimation to produce spatially calibrated velocity maps, vertical and
horizontal flow profiles, per-pixel time series, and validation comparisons against
ideal theoretical data.

Developed at the University of Queensland as part of DECO3801 / Studio 3 (2025),
then refactored and published for portfolio use in 2026.


## What it does

Given a video of flowing water such as a dam spillway, the application:

1. Calibrates spatial scale interactively from a known reference distance the
   user marks by clicking two points on the first frame.
2. Lets the user select up to two regions of interest with per-region gamma
   correction to handle mixed clear-water and turbulent-water conditions.
3. Computes per-frame dense optical flow (Farneback) within each region.
4. Refines optical flow direction using SfS-derived surface normals to suppress
   noise from specular reflections on the water surface.
5. Converts pixel displacements to real-world velocity in metres per second
   using camera perspective correction derived from the known dam geometry.
6. Displays a live quiver plot overlaid on the video frame, showing flow
   direction and magnitude with SfS alignment colour coding.
7. Exports a depth-binned velocity CSV after the live loop ends.
8. Launches an interactive Qt results viewer with heatmap overlays, switchable
   colormaps, hover-to-inspect velocity readout, clickable vertical and
   horizontal profiles, a per-pixel velocity time series, and a comparison
   plot against the reference validation dataset.

The SfS module can also be run standalone to produce depth heatmap videos,
a contour map, a 3D depth surface render, and a depth-profile comparison image.


## Project structure

```
main.py                  Main analysis pipeline: optical flow, SfS refinement, Qt viewer
preprocessing.py         Shared image utilities: rotation, CLAHE, gamma, calibration, ROI
SFSImplementation.py     Standalone Shape-from-Shading depth estimation pipeline
demo.py                  Headless demonstration using a synthetic generated video
requirements.txt         Python package dependencies
```


## Requirements

- Python 3.11 or later
- A video file of flowing fluid. The original dataset (dam spillway footage) is
  protected under an NDA with Professor Hubert Chanson at the University of
  Queensland. Contact the professor directly to request access. Any comparable
  fluid-flow video can be substituted with parameter adjustments.
- An optional validation CSV named `ideal.csv` with columns `Depth m` and
  `Velocity m/s` for comparison against ideal-flow theoretical data.


## Installation

Clone the repository and install dependencies into a virtual environment:

```bash
git clone https://github.com/LAKSHAY-ATREJA/Fluid-Dynamics-Video-Analysis-Application.git
cd Fluid-Dynamics-Video-Analysis-Application

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No API keys or environment variables are required.


## Running the headless demo

The demo generates a synthetic sinusoidal wave video, runs the optical flow and
SfS pipeline without any GUI interaction, and writes three output files to the
current directory:

```bash
python demo.py
```

Expected output:

```
=== Fluid Dynamics Video Analysis — Headless Demo ===

[1/4] Generating synthetic fluid-flow video...
Synthetic video written: demo_input.mp4  (60 frames at 25.0 fps)
[2/4] Running optical flow and SfS depth estimation...
  Processed frame 10/59
  ...
  Processed 59 frames total.
[3/4] Saving velocity heatmap...
  Saved: demo_velocity_heatmap.png
[4/4] Saving velocity profile plot...
  Saved: demo_velocity_profile.png

=== Demo complete ===
  Mean velocity (whole frame):  0.0341 m/s
  99th percentile velocity:     0.1203 m/s

Output files:
  demo_input.mp4
  demo_velocity_heatmap.png
  demo_velocity_profile.png
```

The demo completes in roughly 30 to 90 seconds on a modern laptop, depending
on CPU speed. No display server or Qt is needed.


## Running the main analysis

```bash
python main.py --video /path/to/your_video.MOV
```

Command-line options:

```
--video PATH       Path to the input video file (default: IMGP9471.MOV)
--validation PATH  Path to the reference validation CSV (default: ideal.csv)
--output PATH      Path for the exported velocity CSV (default: velocity_output.csv)
--width-m FLOAT    Known calibration width in metres (default: 12.25)
--rotation INT     Frame rotation angle: 0, 90, 180, or 270 (default: 270)
```

Interactive steps after the application opens:

1. A calibration window shows the first frame. Click two points that span the
   known real-world width (for example, the full width of a dam spillway) and
   press Enter to confirm. Press Escape to skip and use the default estimate of
   0.039 m/px.

2. Draw the first ROI over the upper flow region by clicking and dragging, then
   press Space or Enter to confirm.

3. Adjust the gamma slider for the first ROI. For clear water with visible
   sub-surface texture, values between 4.5 and 5.0 typically work well. Press
   Enter to accept.

4. Draw the second ROI over the lower or turbulent region and press Space or
   Enter.

5. Adjust gamma for the second ROI. For white-water turbulence where specular
   reflections dominate, values between 0.55 and 0.65 typically reduce glare.
   Press Enter to accept.

6. The live quiver plot window shows optical flow vectors in real time. Red
   arrows indicate flow aligned with the SfS downslope direction; orange arrows
   indicate anti-aligned flow that has been partially suppressed. Press Q or
   Escape to end the live loop and open the results viewer.

The results viewer allows switching between JET, TURBO, and VIRIDIS colourmaps,
toggling a 1 m scale bar, hovering over the heatmap to read instantaneous
velocity at any pixel, clicking to pin a time series or a profile plot, and
opening the validation comparison dialog if ideal.csv is present.


## Running the SfS module standalone

```bash
python SFSImplementation.py --video /path/to/your_video.MOV
```

Optional argument:

```
--max-frames N    Maximum number of frames to process (default: all frames)
```

After processing, a calibration step requests four mouse-click points to define
physical scale. The following files are then written to the current directory:

- `sfs_depth_heatmap.mp4` — per-frame SfS relative depth heatmap video
- `physical_depth_heatmap.mp4` — physically scaled depth heatmap video
- `physical_depth_contour_map.png` — contour map of average water depth over
  the selected ROIs
- `average_physical_depth_map.png` — 3D depth surface rendered in metres
- `depth_profile_physical_vs_sfs.png` — horizontal depth profile from the last
  processed frame comparing relative SfS units against physical metres
- `physical_depth_map.csv` — full pixel-level depth map with X, Y, and depth
  columns in metres


## Algorithm notes

Optical flow uses cv2.calcOpticalFlowFarneback with a pyramid scale of 0.5,
three pyramid levels, a window size of 15, and a Gaussian standard deviation of
1.2. SfS depth estimation solves iteratively by gradient descent on the
photometric error between observed intensity and the Lambertian reflectance
model, with Laplacian smoothness regularisation. The depth map gradient defines
a per-pixel downslope direction; the dot product of this direction with the
optical flow unit vector weights the flow magnitude, suppressing motion that
runs counter to the surface slope.

Velocity is calculated as:

    velocity (m/s) = optical_flow_magnitude (px/frame) * fps * mpp

where mpp is metres per pixel from the calibration step. Camera perspective is
corrected using piecewise linear interpolation of view angles measured from the
known dam geometry, applied row by row.


## License

MIT. Built by Lakshay Atreja.
