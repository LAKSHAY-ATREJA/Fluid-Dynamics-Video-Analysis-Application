# Requirements

* Python 3.11 or greater is recommended.
* Libraries: opencv-python, matplotlib, pandas, numpy, PyQt5
* GPU: optional

# External data files needed
Although any video file of flowing fluid can be used with the coreect modifications
* IN this project a video file obtained from Professor Hubert Chanson and is protected under a non-disclosure agreement with this team. You will need to contact the Professor to obtain the necessary approvals to obtain his video file. If you have the file, place it in the root directory of this project.
* "ideal.csv" is a manually created file containing ideal fluid calculations under theoretical ideal conditions provided by Professor Hubert Chanson, subject to the same agreement. This was manually extracted from the Professor's validation dataset into a simpler format. This file contains two columns headed with: "Depth m", "Velocity m/s" and then a series of float values respectively. This is also in the root directory of the project.


# Setup and run

* Ensure all the libraries specified above are installed by running "pip install opencv-python matplotlib pandas numpy PyQt5"
* In your terminal, you can run "python main.py" to run the full program + UI. If at any point you want to quit, press 'Esc' to close each window.
* The first pre-processing step is to select two points that span the width of the dam spillway (at the top) for calibration. Once selected, hit "Enter" or "Space" to continue.
* You now need to select a region-of-interest (ROI). Select this over the darker area of the spillway, but before the white-water parts and hit Space/Enter.
* Refine the 'gamma' slider for this ROI - a good range is around 4.50 to 5.00 for the top ROI, then hit 'Enter'.
* Select the second ROI for the white water section of the spillway, but not all the way to the bottom. Have the bottom of the ROI be roughly 1 or 2 steps down from the where the staircase begins on the LHS of the footage.
* Refine the gamma slider once again, a suitable range is 0.55 to 0.65 for this ROI.
* Let the algorithm run for 10 to 60 seconds and observe the quiver plot.
* Press 'Q' after 10-60 seconds to open the UI and play around with the different features. It should be self-explanatory.

# Shape-from-shading instructions

* In your terminal, run “python SFSImplementation.py” to run the SFS visualisations
* Select the ROI's and gammas as per above
* Let the heatmap plots run over the course of the video, the depths at various locations update live.
* Once the heatmaps have run, a calibration point window appears. Select first the x\_origin which is the top left of the dam where water begins to flow.
* Select the second x\_ref point directly right horizontally from this on the other side of the dam where the water begins to flow.
* Select the y\_origin point which is the top left of the dam where water begins to flow.
* Select the y\_reference point at any point above the step where the stairs begin — do not select any lower, hit Enter.
* Let the algorithm run and a physical depth map will appear, contour map and depth profile comparison between SFS depth and physical depth.
* All these visualisations and a CSV of physical depth data then save down.
