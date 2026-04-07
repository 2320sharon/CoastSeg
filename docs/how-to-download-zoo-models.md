
# Download a Model

If you need to download model, use the main CoastSeg environment, not the `segmentation_workflow` environment.

`conda activate coastseg` or run `pixi shell` in the CoastSeg directory

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