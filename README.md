# Fluid Dynamics Video Analysis Application

A desktop application for measuring water flow velocities in fluid dynamics video footage. The system combines dense optical flow (Farneback method) with Shape-from-Shading (SfS) depth estimation to produce spatially calibrated velocity maps, vertical flow profiles, and comparison plots against theoretical ideal-flow data.

Developed at the University of Queensland as part of DECO3801 / Studio 3 (2025), then refactored and published for portfolio use in 2026.

## What it does

Given a video of flowing water (such as a dam spillway), the application:

1. Calibrates spatial scale from a known reference distance marked by the user
2. Allows selection of up to two regions of interest with per-region gamma correction
3. Computes per-frame dense optical flow within each region
4. Refines optical flow direction using SfS-derived surface normals to reduce noise from specular reflections
5. Converts pixel displacements to real-world velocity (m/s) using camera perspective correction
6. Displays a live quiver plot overlaid on the video frame
7. Launches an interactive Qt results viewer with heatmap overlays, vertical and horizontal velocity profiles, time-series pixel inspection, and validation comparison against a reference dataset

The SfS module (`SFSImplementation.py`) can also be run standalone to produce depth heatmap videos, contour maps, 3D depth surface renders, and a depth-profile comparison PNG.

## Requirements

- Python 3.11 or later
- A video file of flowing fluid. The original dataset (dam spillway footage) is protected under an NDA with Professor Hubert Chanson at the University of Queensland. Contact the professor directly to request access. Any comparable fluid-flow video can be substituted with appropriate parameter adjustments.
- An optional validation CSV (`ideal.csv`) with columns `Depth m` and `Velocity m/s` for comparison against theoretical ideal-flow data.

## Installation

```bash
git clone https://github.com/LAKSHAY-ATREJA/Fluid-Dynamics-Video-Analysis-Application.git
cd Fluid-Dynamics-Video-Analysis-Application

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the main analysis

```bash
python main.py --video /path/to/your_video.MOV
```

Optional arguments:

| Argument       | Default             | Description                              |
|----------------|---------------------|------------------------------------------|
| --video        | IMGP9471.MOV        | Path to the input video file             |
| --validation   | ideal.csv           | Path to the reference validation CSV     |
| --output       | velocity_output.csv | Path for the exported velocity CSV       |
| --width-m      | 12.25               | Known calibration width in metres        |
| --rotation     | 270                 | Frame rotation angle (0/90/180/270)      |

Interactive calibration steps after launch:

1. Click two points spanning the known width (e.g. dam spillway top) and press Enter to confirm
2. Draw the first ROI over the upper flow region and press Space or Enter
3. Adjust the gamma slider for the first ROI (typically 4.5–5.0 for clear water) and press Enter
4. Draw the second ROI over the lower turbulent region and press Space or Enter
5. Adjust the gamma slider for the second ROI (typically 0.55–0.65 for white water) and press Enter
6. Observe the live quiver plot. Press Q after 10–60 seconds to stop and open the results viewer

## Running the SfS module standalone

```bash
python SFSImplementation.py
```

Follow the same ROI/gamma calibration steps. After processing, calibration points for the physical coordinate system are requested, and the following outputs are saved:

- `sfs_depth_heatmap.mp4` — per-frame SfS depth heatmap video
- `physical_depth_heatmap.mp4` — physically scaled depth heatmap video
- `physical_depth_contour_map.png` — contour map of average depth
- `average_physical_depth_map.png` — 3D depth surface render
- `depth_profile_physical_vs_sfs.png` — depth profile comparison

## Project structure

```
main.py                  Main analysis application (optical flow + SfS + Qt UI)
preprocessing.py         Image utilities: rotation, CLAHE, gamma, calibration, ROI selection
SFSImplementation.py     Standalone Shape-from-Shading depth estimation pipeline
requirements.txt         Python dependencies
```

## Algorithm notes

Optical flow is computed with `cv2.calcOpticalFlowFarneback`. SfS refines the flow direction by computing surface normals from an iteratively solved depth map (gradient descent on photometric error with Laplacian smoothness regularisation). Velocities are corrected for camera perspective using piecewise linear angular interpolation derived from the known dam geometry. The results viewer allows switching between JET, TURBO, and VIRIDIS colourmaps and toggling a scale bar.

## License

MIT. Built by Lakshay Atreja.
