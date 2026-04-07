# Install the Zoo Workflow

There are two ways to install the CoastSeg zoo workflow:

- Pixi
- Conda

Pixi is the recommended option.

## Install with Pixi (Recommended)

Pixi is the recommended way to install the zoo workflow.

1. Open a terminal.
2. Go to the `segmentation_workflow` folder.
    ```bash
    cd CoastSeg/segmentation_workflow
    ```
3. Create and activate the Pixi environment.
    ```bash
    pixi shell
    ```

4. Wait for the environment to finish installing and activating.
5. Confirm the environment is active.
    - You will know it is active when you see `(segmentation_workflow)` in the terminal.
    ```bash
    (segmentation_workflow) C:\CoastSeg\segmentation_workflow
    ```
6. Exit the Pixi environment.

```bash
exit
```

## Install with Conda

Conda also works, but we do not recommend it.

!!! warning
    The Conda workflow requires you to run the models with the script `run_zoo_segmentation_models.py` in the folder `segmentation_workflow` and then run the script `3_zoo_workflow_extract_shorelines.py` in the main CoastSeg environment to extract shorelines from the segmentations.
1. Open Miniforge.
2. Go to the `CoastSeg/segmentation_workflow` folder
    - Remember to use the location where you installed CoastSeg on your computer.
    ```bash
    cd <CoastSeg_location>
    cd segmentation_workflow
    ```

3. Install the Conda environment using the instructions in `segmentation_workflow`.

```bash
conda install -f conda.yml
```

This workflow works fine, but Pixi is the recommended setup for the zoo workflow.