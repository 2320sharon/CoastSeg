import os
from coastseg import coastseg_logs
from coastseg import zoo_model
from coastseg.tide_correction import compute_tidal_corrections
from coastseg import file_utilities

# Script instructions:
#   This script only extracts shorelines from existing segmentation outputs.
#   It does NOT run segmentation models.
#
#   Before running this script, first run:
#     segmentation_workflow/run_zoo_segmentation_models.py
#   Follow the setup/run steps in:
#     segmentation_workflow/how_to_run_models.md
#
#   Then set `session_path` below to the folder created by
#   run_zoo_segmentation_models.py that contains the segmentation results
#   (for example, the folder with *_predseg.png files and summary metadata).

# This script uses CoastSeg's zoo-model shoreline extraction workflow.
# It processes one ROI/session at a time.


# 1. ENTER THE DIRECTORY WHERE THE SEGMENTATION MODEL PREDICTIONS ARE STORED
# ---------------------------
# - Enter location of directory containing the segmentations from run_zoo_segmentation_models.py
# - Example path : 'CoastSeg\sessions\model_predictions'
session_path = r""

# Extract Shoreline Settings
settings = {
    "min_length_sl": 100,  # minimum length (m) of shoreline perimeter to be valid
    "max_dist_ref": 500,  # maximum distance (m) from reference shoreline to search for valid shorelines. This detrmines the width of the buffer around the reference shoreline
    "cloud_thresh": 0.5,  # threshold on maximum cloud cover (0-1). If the cloud cover is above this threshold, no shorelines will be extracted from that image
    "dist_clouds": 100,  # distance(m) around clouds where shoreline will not be mapped
    "min_beach_area": 50,  # minimum area (m^2) for an object to be labelled as a beach
    "sand_color": "default",  # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    "apply_cloud_mask": True,  # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
}


# 2. Save the settings to the model instance
# -----------------
# Create an instance of the zoo model to run the model predictions
zoo_model_instance = zoo_model.Zoo_Model()
# save the settings to the model instance
zoo_model_instance.set_settings(**settings)


# OPTIONAL: If you have a transects and shoreline file, you can extract shorelines from the zoo model outputs
transects_path = ""  # path to the transects geojson file (optional, default will be loaded if not provided)
shoreline_path = ""  # path to the shoreline geojson file (optional, default will be loaded if not provided)
shoreline_extraction_area_path = (
    ""  # path to the shoreline extraction area geojson file (optional)
)

# 3. Extract shorelines from existing segmentations
# -------------------------------------
# This step does not run segmentation models; it only extracts shorelines.
# First run segmentation_workflow/run_zoo_segmentation_models.py.
# Set `session_path` to that output folder (the folder with *_predseg.png files).
zoo_model_instance.extract_shorelines(
    session_path=session_path,
    shoreline_path=shoreline_path,
    transects_path=transects_path,
    shoreline_extraction_area_path=shoreline_extraction_area_path,
)

# 4. OPTIONAL: Run Tide Correction
# ------------------------------------------
# Tide Correction (optional)
# WARNING: Before running this snippet, you must download the tide model to the CoastSeg/tide_model folder.
# WE RECOMMEND USING FES2022.
#
# Tutorial on How to Download the Tide Model:
# https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model
#
# The Tide Model must be downloaded to CoastSeg/tide_model.
# Two Tide Models are available: 'FES2014' or 'FES2022'.
#
# Parameters:
beach_slope = 0.02  # Slope of the beach (m/m)
reference_elevation = (
    0  # Reference elevation (m, relative to user-specified vertical datum)
)
tides_file = ""  # (Optional) Enter the full path to the CSV file containing the tide data if you don't want to use the tide model. See accepted formats : https://satelliteshorelines.github.io/CoastSeg/tide-file-format/
slopes_file = ""  # (Optional) Enter the full path to the CSV file containing the beach slopes if you don't want to use a constant slope. See accepted formats: https://satelliteshorelines.github.io/CoastSeg/slope-file-format/
if slopes_file:
    beach_slope = slopes_file

# UNCOMMENT THESE 2 LINES TO RUN THE TIDE CORRECTION
# roi_id = file_utilities.get_ROI_ID_from_session(session_name) # read ROI ID from the config.json file found in the extracted shoreline session directory
# compute_tidal_corrections(session_name, [roi_id], beach_slope, reference_elevation,model='FES2022',tides_file=tides_file)
