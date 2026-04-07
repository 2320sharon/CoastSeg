# How to Use the Zoo Notebook
- ⚠️ Ensure you have downloaded data from Google Earth Engine before running Zoo models
- ⚠️ Zoo notebook runs one region of interest (ROI) at a time

# Before you begin 
Follow the installation instructions in [Install the Zoo Workflow](install-zoo-workflow.md) before using the zoo workflow models.

# Phase 1: Install the Models 
1. Activate the CoastSeg Environment `conda activate CoastSeg`
2. Run the `SDS_zoo_classifier.ipynb` notebook and download a model using the download button
	- Alternatively,  use the `download_zoo_model.py` script to download one of the available zoo models. Follow this guide for more details on how to download models using the script [How to Download Models](how-to-download-zoo-models.md)
3. Validate the model you downloaded is at `CoastSeg/models`

# Phase 2: Run the models

## 1. Activate the environment

Choose one option.

### Pixi (Recommended)

```bash
cd <coastseg_location>
cd segmentation_workflow
pixi shell
```

`pixi shell` installs the environment on first run, then activates it.
- This command is like running `conda activate` & `conda install` at the same time

### Conda

```bash
cd segmentation_workflow
conda activate segmentation_workflow
```

## 2. Run the Models

First, move into the workflow directory and make sure the environment is active.

```bash
cd segmentation_workflow
```

Activate the environment with either `pixi shell` or `conda activate segmentation_workflow`.

### CLI Command

Defaults:

- `--implementation BEST`
- GPU is used if available

```bash
python run_zoo_segmentation_models.py \
	-i "<input_dir>" \
	-o "<output_dir>" \
	-m "<model_dir>" \
	[--implementation BEST|ENSEMBLE] \
	[--overwrite] \
	[--cpu-only]
```

### Parameters

- `-i`, `--input-dir`: Folder containing input images (`jpg`, `png`, or `tif`).
- `-o`, `--output-dir`: Destination for prediction masks and `segmentation_summary.json`.
- `-m`, `--model`: Model directory containing the `.h5` weights and `.json` config files.

### Example

```bash
python run_zoo_segmentation_models.py \
	-i "C:\path\to\jpg_files" \
	-o "C:\path\to\predictions" \
	-m "C:\path\to\model_dir"
```

### Real Example

```bash
python run_zoo_segmentation_models.py \
	-i "C:\CoastSeg\data\ID_rnv2_datetime01-16-26__03_50_41\jpg_files\preprocessed\RGB" \
	-o "C:\CoastSeg\sessions\model_outputs_session" \
	-m "C:\CoastSeg\models\global_segformer_RGB_4class_14036903"
```

- This example will run the `global_segformer_RGB_4class_14036903` on the images in the directory `C:\CoastSeg\data\ID_rnv2_datetime01-16-26__03_50_41\jpg_files\preprocessed\RGB` and save the segmentations to the directory `"C:\CoastSeg\sessions\model_outputs_session`

## Outputs

The workflow writes all outputs to `<output_dir>`.

For each input image, it creates:

- `<image_stem>_predseg.png`: A colorized segmentation mask.
- `<image_stem>_res.npz`: A compressed NumPy file containing the predicted labels and metadata.

It also writes these run-level files:

- `model_settings.json`: The segmentation settings used for the run, including the implementation mode, model type, and GPU setting.
- `model_info.json`: Metadata about the model, including the model directory, class names, and water-class indices.
- `segmentation_summary.json`: A summary of the run, including the total number of images processed, masks written, NPZ files written, skipped files, and any failures.

The output folder keeps the same relative folder structure as the input folder, with each predicted mask saved as `_predseg.png` and each companion data file saved as `_res.npz`.


# Phase 3: Extract Shorelines from the Segmentations

1. Deactivate the `segmentation_workflow` environment
2. Go back to the CoastSeg environment `cd ..`
3. Run the script `3_zoo_workflow_extract_shorelines.py`
- Follow the instructions in the script by opening it in a code editor like VSCode