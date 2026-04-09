import os
from coastseg import coastseg_logs
from coastseg import zoo_model
from coastseg.tide_correction import compute_tidal_corrections
from coastseg import file_utilities

# Script instructions:
#   This script is for Pixi users only.
#   It uses the segmentation workflow pixi environment managed in CoastSeg/segmentation_workflow.
#
#   Before running this script, install the Pixi environment:
#     1) cd segmentation_workflow
#     2) pixi install
#
#   Then run this script from the CoastSeg repository root.
#
#   If you do NOT use Pixi, use this script instead:
#     3_zoo_workflow_extract_shorelines.py

# The Zoo Model is a machine learning model that can be used to extract shorelines from satellite imagery.
# This script will only run a single ROI at a time. If you want to run multiple ROIs, you will need to run this script multiple times.

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


# The model can be run using the following settings:
model_setting = {
    "img_type": "RGB",  # Available options: "RGB", "MNDWI", "NDWI". Make sure the model name is compatible with the image type
    "use_GPU": "0",  # 0 or 1. 0 means no GPU
    "implementation": "BEST",  # BEST or ENSEMBLE. BEST is recommended for most users. ENSEMBLE can be used to run multiple models and ensemble the results
    "model_type": "global_segformer_RGB_4class_14036903",  # model name from the zoo
    "apply_segmentation_filter": False,  # apply segmentation filter to the model outputs to sort them into good or bad
}

# 1. Set the User configuration Settings
# ---------------------------
# a. ENTER THE NAME OF THE SESSION TO SAVE THE MODEL PREDICTIONS TO
session_name = "coreg_session_demo"
# b. ENTER THE DIRECTORY WHERE THE INPUT IMAGES ARE STORED
# - Enter location of directory containing RGB imagery within coastseg. Note this is the RGB folder within the CoastSeg/data directory
# - Example path t : 'CoastSeg\data\ID_zyh1_datetime06-11-24__03_02_55\coregistered\jpg_files\preprocessed\RGB'
input_directory = r""


# 2. Save the settings to the model instance
# -----------------
# Create an instance of the zoo model to run the model predictions
zoo_model_instance = zoo_model.Zoo_Model()
# save settings to the zoo model instance
settings.update(model_setting)
# save the settings to the model instance
zoo_model_instance.set_settings(**settings)


# OPTIONAL: If you have a transects and shoreline file, you can extract shorelines from the zoo model outputs
transects_path = r""  # path to the transects geojson file (optional, default will be loaded if not provided)
shoreline_path = r""  # path to the shoreline geojson file (optional, default will be loaded if not provided)
shoreline_extraction_area_path = (
    ""  # path to the shoreline extraction area geojson file (optional)
)

# 3. Run the model and extract shorelines
# -------------------------------------
zoo_model_instance.run_model_and_extract_shorelines(
    input_directory,
    session_name=session_name,
    shoreline_path=shoreline_path,
    transects_path=transects_path,
    shoreline_extraction_area_path=shoreline_extraction_area_path,
    coregistered=True,
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
