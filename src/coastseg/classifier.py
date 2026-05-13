import glob
import os
import shutil
from typing import List, Optional, Tuple

import pooch
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from coastseg import common, file_utilities

# Some of these functions were originally written by Mark Lundine and have been modified for this project.


MODEL_CONFIG = {
    "rgb": {
        "model_name": "ImageRGBClassifier",
        "url": "https://huggingface.co/2320sharon/ImageRGBClassifier/resolve/main/best.onnx?download=true",
        "sha256": "sha256:aba43106e5a38e978cccdc852c8379a9de41b7d0d6dd9d323012106e4e7a965a",
    },
    "segmentation": {
        "model_name": "ShorelineFilter",
        "url": "https://huggingface.co/2320sharon/ShorelineFilter/resolve/main/best_seg.onnx?download=true",
        "sha256": "sha256:e0edd406427308a854da9c3f1c85571dc0c19ac481a6e25cc128d3f3c8c0b04f",
    },
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0
    negative = ~positive
    out = np.empty_like(x, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)
    return out


def _is_nchw(input_shape: List[object]) -> bool:
    """
    Determine if the input shape corresponds to NCHW format.
    Checks if the input shape has 4 dimensions and if the channel dimension is in the expected position for NCHW (either 1 or 3). Returns True if it is NCHW
    and False otherwise.

    Args:
        input_shape (List[object]): The shape of the input tensor as a list of dimensions.
    Returns:
        bool: True if the input shape is in NCHW format, False otherwise.
    Example:
        >>> _is_nchw([None, 3, 128, 128])  # NCHW format with 3 channels
        True
        >>> _is_nchw([None, 128, 128, 3])  # NHWC format
        False
        >>> _is_nchw([None, 1, 128, 128])  # NCHW format with 1 channel
        True
        >>> _is_nchw([None, 128, 128, 1])  # NHWC format with 1 channel
        False
        >>> _is_nchw([None, 128, 128])  # Invalid shape
        False
    """
    # checks if image tensors are 4D eg. batch + channels + height + width
    if len(input_shape) != 4:
        return False

    # checks if the second dimension is the channel dimension
    channel_axis = input_shape[1]
    # checks if the channel dimension is either 1 or 3, which are common for grayscale and RGB images respectively
    if isinstance(channel_axis, int) and channel_axis in (1, 3):
        return True

    return False


def _prepare_input_tensor(
    image_path: str,
    image_size: Tuple[int, int],
    channels: int,
    nchw: bool,
) -> np.ndarray:
    """Load an image and format it as float32 batch tensor for ONNX inference."""
    image = Image.open(image_path)
    if channels == 1:
        image = image.convert("L")
    elif channels == 3:
        image = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    resample = getattr(Image, "Resampling", Image).BILINEAR  # type: ignore[attr-defined]
    image = image.resize(image_size, resample)
    image_array = np.asarray(image, dtype=np.float32)

    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)

    if nchw:
        image_array = np.transpose(image_array, (2, 0, 1))

    return np.expand_dims(image_array, axis=0)


def _infer_scores(
    model_path: str,
    image_paths: List[str],
    image_size: Tuple[int, int],
    expected_channels: int,
) -> List[float]:
    """Runs ONNX inference on a list of images and returns sigmoid scores.

    Loads an ONNX model from ``model_path``, infers the expected input layout
    (NCHW or NHWC) from the first model input, prepares each image to match the
    model's input requirements, and runs inference on CPU. The raw model output
    is converted to a probability-like score by applying a sigmoid and taking
    the first scalar value.

    Args:
        model_path: Path to the ONNX model file.
        image_paths: Paths to input image files to score.
        image_size: Target image size as ``(width, height)`` for preprocessing.
        expected_channels: Fallback number of image channels to use when the
            model input shape does not specify a concrete channel dimension.

    Returns:
        A list of sigmoid-transformed scores, one for each image in
        ``image_paths``.

    Raises:
        ValueError: If the inferred or fallback channel count is not supported.
            Only 1-channel and 3-channel image inputs are supported.
        onnxruntime.capi.onnxruntime_pybind11_state.Fail: If the ONNX model
            cannot be loaded or inference fails.
        FileNotFoundError: If an image path or the model path does not exist,
            depending on the behavior of the underlying runtime or image-loading
            code.

    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[
        0
    ]  # get the metadata for the first input layer of the ONNX model
    input_name = input_meta.name  # get the name of the input layer for the ONNX model
    input_shape = list(
        input_meta.shape
    )  # get the expected input shape for the model (e.g., [None, 3, 128, 128] or [None, 128, 128, 3])
    nchw = _is_nchw(input_shape)

    channel_index = (
        1 if nchw else -1
    )  # channel index is the last dimension if not nchw and the second dimension if nchw
    model_channels = input_shape[channel_index]
    channels = expected_channels
    if isinstance(model_channels, int) and model_channels > 0:
        channels = model_channels
    if channels not in (1, 3):
        raise ValueError(
            f"Unsupported input shape for ONNX model {model_path}: {input_shape}"
        )

    model_scores: List[float] = []
    for im_path in image_paths:
        input_tensor = _prepare_input_tensor(im_path, image_size, channels, nchw)
        logits = np.asarray(session.run(None, {input_name: input_tensor})[0])
        score = float(
            sigmoid(logits).reshape(-1)[0]
        )  # take the first scalar value from the sigmoid output
        model_scores.append(score)

    return model_scores


def filter_segmentations(
    session_path: str,
    threshold: float = 0.40,
) -> str:
    """
    Sort model output files into "good" and "bad" folders based on the segmentation classifier.

    Uses a trained segmentation classifier model to evaluate shoreline segmentation images
    and sorts them into "good" and "bad" directories based on quality scores.

    Args:
        session_path (str): The path to the session directory containing the model output files.
        threshold (float): Classification threshold for determining good vs bad segmentations.
            Images with scores >= threshold are classified as "good". Defaults to 0.40.

    Returns:
        str: The path to the "good" folder containing the sorted model output files.

    Raises:
        FileNotFoundError: If no model output files are found at the session path or if
            the classification process fails to create the good folder.

    Example:
        >>> good_path = filter_segmentations("/path/to/session", threshold=0.5)
        >>> print(good_path)
        /path/to/session/good
    """
    segmentation_classifier = get_segmentation_classifier()
    good_path = os.path.join(session_path, "good")
    csv_path, good_path, bad_path = run_inference_segmentation_classifier(
        segmentation_classifier,
        session_path,
        session_path,
        good_path=good_path,
        threshold=threshold,
    )
    # if the good folder does not exist then this means the classifier could not find any png files at the session path and something went wrong
    if not os.path.exists(good_path):
        raise FileNotFoundError(
            f"No model output files found at {session_path}. Shoreline Filtering failed."
        )
    return good_path


def move_matching_files(
    input_image_path: str, search_string: str, file_exts: List[str], target_dir: str
) -> None:
    """
    Move files matching the given search string and file extensions to the target directory.

    Searches for files in the same directory as the input image that contain the search string
    and have one of the specified file extensions. Moves all matching files to the target directory.

    Args:
        input_image_path (str): Path to the original input image.
        search_string (str): The string to look for in filenames.
        file_exts (List[str]): List of file extensions to match (e.g., ['.jpg', '.jpeg', '.png']).
        target_dir (str): Directory where matching files should be moved.

    Returns:
        None

    Example:
        >>> input_image_path = 'C:/path/to/image.jpg'
        >>> search_string = '2021-01-01'
        >>> file_exts = ['.jpg', '.jpeg', '.png']
        >>> target_dir = 'C:/path/to/target_dir'
        >>> move_matching_files(input_image_path, search_string, file_exts, target_dir)
        # All files matching the search string and file extensions will be moved to target_dir
    """
    for ext in file_exts:
        # Create the search pattern
        pattern = os.path.join(
            os.path.dirname(input_image_path), f"*{search_string}*{ext}"
        )
        matching_files = glob.glob(pattern)
        for matching_file in matching_files:
            if os.path.exists(matching_file):
                output_image_path = os.path.join(
                    target_dir, os.path.basename(matching_file)
                )
                shutil.move(matching_file, output_image_path)


def sort_images_with_model(
    input_directory: str,
    type: str = "rgb",
    threshold: float = 0.40,
    print_status: bool = False,
) -> None:
    """
    Sort images in a directory using a trained good/bad image classifier model.

    Bad images are moved to a 'bad' subdirectory while good images remain in the original directory.
    The classification is based on the model's confidence score compared to the threshold.

    Args:
        input_directory (str): The directory containing the images to be classified.
            Should contain jpg, png, or jpeg files.
        type (str): The type of model to use. Options are 'rgb' or 'gray'.
            The RGB model is used for color images and the gray model is used for
            grayscale images or RGB images. Defaults to 'rgb'.
        threshold (float): Threshold on sigmoid of model output. Images with scores >= threshold
            are classified as good (e.g., 0.6 means mark images as good if model output is >= 0.6,
            or 60% sure it's a good image). Defaults to 0.40.
        print_status (bool): If True, print a summary showing that sorting finished,
            where the classification CSV was saved, and where bad files were moved.
            Defaults to False.

    Returns:
        None

    Example:
        >>> sort_images_with_model(
        ...     input_directory='C:/Coastseg/data/ID_<roi_id>_datetime<date>/jpg_files/preprocessed/RGB',
        ...     type='rgb',
        ...     threshold=0.40,
        ...     print_status=True,
        ... )
        # Bad images will be moved to a 'bad' subdirectory
    """
    classifier_path = get_image_classifier(type)
    csv_path = os.path.join(input_directory, "image_classification_results.csv")
    bad_path = os.path.join(input_directory, "bad")

    if type.lower() == "rgb":
        csv_path = run_inference_rgb_image_classifier(
            classifier_path, input_directory, input_directory, threshold=threshold
        )
    else:
        csv_path = run_inference_gray_image_classifier(
            classifier_path, input_directory, input_directory, threshold=threshold
        )

    if print_status:
        print(f"Files were sorted in {input_directory}")
        print(f"Classification CSV saved to {csv_path}")
        print(f"Bad files were moved to {bad_path}")


def sort_images(
    inference_df_path: str,
    output_folder: str,
    good_path: str = "",
    bad_path: str = "",
    threshold: float = 0.40,
    file_exts: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Sort images into good and bad folders based on model inference results.

    Uses model results from a CSV file to sort images and optionally matching files with
    specified extensions into good and bad directories based on the threshold.This will sort the npz files as well as matching files with the extensions in ['.jpg', '.jpeg', '.png'] from the inference results
     into good and bad folders based on the threshold

    Args:
        inference_df_path (str): Path to the CSV file containing model results.
        output_folder (str): Path to the directory containing the inference images.
        good_path (str): Path to directory for good images. If empty, creates 'good'
            subdirectory in output_folder. Defaults to "".
        bad_path (str): Path to directory for bad images. If empty, creates 'bad'
            subdirectory in output_folder. Defaults to "".
        threshold (float): Threshold of model output for classification. Images with scores
            >= threshold are classified as good. Defaults to 0.40.
        file_exts (Optional[List[str]]): List of file extensions to match when moving
            associated files to good or bad directories. Defaults to None.

    Returns:
        Tuple[str, str]: A tuple containing (good_path, bad_path) - the paths to directories
            containing the good and bad images respectively.

    Example:
        >>> inference_df_path = 'C:/path/to/inference_results.csv'
        >>> output_folder = 'C:/path/to/output_folder'
        >>> file_exts = ['.jpg', '.jpeg', '.png']
        >>> good_path, bad_path = sort_images(inference_df_path, output_folder,
        ...                                   threshold=0.40, file_exts=file_exts)
        >>> print(f"Good images: {good_path}, Bad images: {bad_path}")
        Good images: C:/path/to/output_folder/good, Bad images: C:/path/to/output_folder/bad
    """
    if not file_exts:
        file_exts = []
    if not good_path:
        good_path = os.path.join(output_folder, "good")
    if not bad_path:
        bad_path = os.path.join(output_folder, "bad")

    dirs = [output_folder, bad_path, good_path]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df["im_paths"].iloc[i]
        im_name = os.path.basename(input_image_path)

        if inference_df["model_scores"].iloc[i] < threshold:
            date = common.extract_date_from_filename(im_name)
            # for each file extentsion in the list get the matching file that match the im_name date
            move_matching_files(input_image_path, date, file_exts, bad_path)
            output_image_path = os.path.join(bad_path, im_name)
            shutil.move(input_image_path, output_image_path)
        else:  # if it was higher than the threshold it was a good image and should be moved to the good directory
            date = common.extract_date_from_filename(im_name)
            move_matching_files(input_image_path, date, file_exts, good_path)
            output_image_path = os.path.join(good_path, im_name)
            shutil.move(input_image_path, output_image_path)
    return good_path, bad_path


def run_inference_rgb_image_classifier(
    path_to_model_ckpt: str,
    path_to_inference_imgs: str,
    output_folder: str,
    csv_path: str = "",
    threshold: float = 0.40,
) -> str:
    """
    Run a trained RGB image classifier on images and sort them into good or bad folders.

    Classifies images as either good or bad using a trained model, saves results to CSV,
    and sorts images into appropriate folders based on the classification threshold.

    Args:
        path_to_model_ckpt (str): Path to the ONNX model file.
        path_to_inference_imgs (str): Path to the folder containing images to classify.
            Should contain '.jpg', '.jpeg', or '.png' files.
        output_folder (str): Path to save outputs to.
        csv_path (str): Path to save CSV results. If not provided, results will be saved to
            output_folder/image_classification_results.csv. Defaults to "".
        threshold (float): Threshold on sigmoid of model output for classification.
            Images with scores >= threshold are classified as good. Defaults to 0.40.

    Returns:
        str: Path to the CSV file containing the classification results.

    Example:
        >>> csv_path = run_inference_rgb_image_classifier(
        ...     path_to_model_ckpt="model.onnx",
        ...     path_to_inference_imgs="/path/to/images",
        ...     output_folder="/path/to/output",
        ...     threshold=0.6
        ... )
        >>> print(f"Results saved to: {csv_path}")
        Results saved to: /path/to/output/image_classification_results.csv
    """
    if not csv_path:
        csv_path = os.path.join(output_folder, "image_classification_results.csv")

    os.makedirs(output_folder, exist_ok=True)

    image_size = (128, 128)
    types = ("*.jpg", "*.jpeg", "*.png")
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = _infer_scores(
        path_to_model_ckpt,
        im_paths,
        image_size=image_size,
        expected_channels=3,
    )
    ##save results to a csv
    df = pd.DataFrame(
        {
            "im_paths": im_paths,
            "model_scores": model_scores,
            "threshold": np.full(len(im_paths), threshold),
        }
    )

    df.to_csv(csv_path, index=False)
    sort_images(csv_path, output_folder, good_path=output_folder, threshold=threshold)
    return csv_path


def run_inference_gray_image_classifier(
    path_to_model_ckpt: str,
    path_to_inference_imgs: str,
    output_folder: str,
    csv_path: str = "",
    threshold: float = 0.40,
) -> str:
    """
    Run a trained grayscale image classifier on images and sort them into good or bad folders.

    Classifies images as either good or bad using a trained grayscale model, saves results to CSV,
    and sorts images into appropriate folders based on the classification threshold.

    Args:
        path_to_model_ckpt (str): Path to the ONNX model file.
        path_to_inference_imgs (str): Path to the folder containing images to classify.
            Should contain '.jpg', '.jpeg', or '.png' files.
        output_folder (str): Path to save outputs to.
        csv_path (str): Path to save CSV results. If not provided, results will be saved to
            output_folder/image_classification_results.csv. Defaults to "".
        threshold (float): Threshold on sigmoid of model output for classification.
            Images with scores >= threshold are classified as good. Defaults to 0.40.

    Returns:
        str: Path to the CSV file containing the classification results.

    Example:
        >>> csv_path = run_inference_gray_image_classifier(
        ...     path_to_model_ckpt="model.onnx",
        ...     path_to_inference_imgs="/path/to/images",
        ...     output_folder="/path/to/output",
        ...     threshold=0.6
        ... )
        >>> print(f"Results saved to: {csv_path}")
        Results saved to: /path/to/output/image_classification_results.csv
    """
    if not csv_path:
        csv_path = os.path.join(output_folder, "image_classification_results.csv")

    os.makedirs(output_folder, exist_ok=True)
    image_size = (128, 128)
    types = ("*.jpg", "*.jpeg", "*.png")
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = _infer_scores(
        path_to_model_ckpt,
        im_paths,
        image_size=image_size,
        expected_channels=1,
    )
    ##save results to a csv
    df = pd.DataFrame(
        {
            "im_paths": im_paths,
            "model_scores": model_scores,
            "threshold": np.full(len(im_paths), threshold),
        }
    )
    df.to_csv(csv_path, index=False)
    sort_images(csv_path, output_folder, good_path=output_folder, threshold=threshold)
    return csv_path


def _get_onnx_model(model_key: str) -> str:
    """Download and return a configured ONNX model path.

    Args:
        model_key (str): Model identifier in ``MODEL_CONFIG``.
            This is either "rgb" for the RGB image classifier or "segmentation" for the shoreline segmentation quality classifier.

    Returns:
        str: Full path to the ONNX model file.

    Raises:
        ValueError: If ``model_key`` is not configured.
        FileNotFoundError: If the downloaded ONNX model file is missing.
    """
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model key: {model_key}")

    config = MODEL_CONFIG[model_key]
    model_directory = file_utilities.create_directory(
        common.get_downloaded_models_dir(), config["model_name"]
    )

    # Extract filename from URL (before query params)
    filename = config["url"].split("/")[-1].split("?")[0]

    downloaded_path = pooch.retrieve(
        url=config["url"],
        known_hash=MODEL_CONFIG[model_key].get("sha256"),
        progressbar=True,
        path=model_directory,
        fname=filename,
    )
    if os.path.exists(downloaded_path):
        return downloaded_path

    raise FileNotFoundError(f"No ONNX model found at {downloaded_path}")


def get_image_classifier(type: str = "rgb") -> str:
    """
    Download and return the path to a pre-trained image classifier model.

    Downloads the appropriate RGB or grayscale image classifier model from GitHub
    and returns the local file path for inference.

    Args:
        type (str): Type of model to download. Options are 'rgb' for color images
            or 'gray' for grayscale images. Defaults to 'rgb'.

    Returns:
        str: Full path to the ONNX model file.

    Raises:
        FileNotFoundError: If no expected ONNX model file exists.

    Example:
        >>> rgb_model_path = get_image_classifier(type='rgb')
        >>> print(f"RGB model downloaded to: {rgb_model_path}")
        RGB model path: /path/to/downloaded_models/ImageRGBClassifier/best.onnx
    """
    model_type = type.lower()
    if model_type not in ("rgb"):
        raise ValueError("type must be either 'rgb'")
    return _get_onnx_model(model_type)


def get_segmentation_classifier() -> str:
    """
    Download and return the path to a pre-trained segmentation classifier model.

    Downloads the segmentation quality classifier model from GitHub that can evaluate
    the quality of shoreline segmentation results and classify them as good or bad.

    Returns:
        str: Full path to the segmentation classifier ONNX model file.

    Raises:
        FileNotFoundError: If no expected ONNX model file exists.

    Example:
        >>> model_path = get_segmentation_classifier()
        >>> print(f"Segmentation classifier downloaded to: {model_path}")
        Segmentation classifier path: /path/to/downloaded_models/ShorelineFilter/best_seg.onnx
    """
    return _get_onnx_model("segmentation")


def run_inference_segmentation_classifier(
    path_to_model_ckpt: str,
    path_to_inference_imgs: str,
    output_folder: str,
    csv_path: str = "",
    good_path: str = "",
    bad_path: str = "",
    threshold: float = 0.10,
) -> Tuple[str, str, str]:
    """
    Run a trained segmentation classifier on images and sort them into good or bad folders.

    Classifies segmentation images as either good or bad using a trained model, saves results
    to CSV, and sorts images (including associated .npz files) into appropriate folders based
    on the classification threshold.

    Args:
        path_to_model_ckpt (str): Path to the ONNX model file.
        path_to_inference_imgs (str): Path to the folder containing segmentation images to classify.
            Should contain '.jpg', '.jpeg', or '.png' files.
        output_folder (str): Path to save outputs to.
        csv_path (str): Path to save CSV results. If not provided, results will be saved to
            output_folder/segmentation_classification_results.csv. Defaults to "".
        good_path (str): Path to directory for good segmentations. If empty, creates 'good'
            subdirectory in output_folder. Defaults to "".
        bad_path (str): Path to directory for bad segmentations. If empty, creates 'bad'
            subdirectory in output_folder. Defaults to "".
        threshold (float): Threshold on sigmoid of model output for classification.
            Images with scores >= threshold are classified as good. Defaults to 0.10.

    Returns:
        Tuple[str, str, str]: A tuple containing (csv_path, good_path, bad_path) -
            the paths to the CSV results file and directories containing good and bad images.

    Example:
        >>> csv_path, good_path, bad_path = run_inference_segmentation_classifier(
        ...     path_to_model_ckpt="segmentation_model.onnx",
        ...     path_to_inference_imgs="/path/to/segmentations",
        ...     output_folder="/path/to/output",
        ...     threshold=0.3
        ... )
        >>> print(f"Results: {csv_path}")
        >>> print(f"Good segmentations: {good_path}")
        >>> print(f"Bad segmentations: {bad_path}")
    """
    os.makedirs(output_folder, exist_ok=True)

    if not good_path:
        good_path = os.path.join(output_folder, "good")
    if not bad_path:
        bad_path = os.path.join(output_folder, "bad")

    dirs = [output_folder, bad_path, good_path]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    image_size = (512, 512)
    types = ("*.jpg", "*.jpeg", "*.png")
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))

    # If not files exist return the good and bad paths. This is assuming the files were previously sorted
    if im_paths == []:
        return csv_path, good_path, bad_path

    model_scores = _infer_scores(
        path_to_model_ckpt,
        im_paths,
        image_size=image_size,
        expected_channels=3,
    )
    ##save results to a csv
    df = pd.DataFrame(
        {
            "im_paths": im_paths,
            "model_scores": model_scores,
            "threshold": np.full(len(im_paths), threshold),
        }
    )

    if not csv_path:
        csv_path = os.path.join(
            output_folder, "segmentation_classification_results.csv"
        )

    df.to_csv(csv_path, index=False)
    good_path, bad_path = sort_images(
        csv_path,
        output_folder,
        good_path=good_path,
        bad_path=bad_path,
        threshold=threshold,
        file_exts=["npz"],
    )
    return csv_path, good_path, bad_path
