# How to Run the Segmentation Workflow

## 1. Install the Environment

Choose one option.

### Pixi (Recommended)

```bash
cd segmentation_workflow
pixi shell
```

`pixi shell` installs the environment on first run, then activates it.
- This command is like running `conda activate` & `conda install` at the same time

### Conda

```bash
cd segmentation_workflow
conda env create -f conda.yml
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

## 3. Download a Model

If you need a model, use the main CoastSeg environment, not the `segmentation_workflow` environment.

List available models:

```bash
python download_zoo_model.py --list-models
```

Download a specific model:

```bash
python download_zoo_model.py --model-name <name>
```

Recommended model:

```bash
python download_zoo_model.py --model-name global_segformer_RGB_4class_14036903
```

This downloads the global RGB SegFormer model to the default location:

`CoastSeg/models/global_segformer_RGB_4class_14036903`

Only the best weights for the model are downloaded.

