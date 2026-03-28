from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

tf = importlib.import_module("tensorflow")

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    transformers_logging = importlib.import_module("transformers.utils.logging")

    transformers_logging.set_verbosity_error()
except Exception:
    pass


FAILURE_PREFIX = "SEGMENTATION_FAILURE: "
PROGRESS_PREFIX = "SEGMENTATION_PROGRESS: "


def median_smooth(img: np.ndarray, size: int = 15) -> np.ndarray:
    """Apply a median filter to an image.

    Args:
        img (np.ndarray): Input image array.
        size (int): Median filter window size.

    Returns:
        np.ndarray: Smoothed image array.
    """
    if img.ndim == 3:
        return np.stack(
            [_median_filter_2d(img[..., i], size=size) for i in range(img.shape[2])],
            axis=-1,
        )
    return _median_filter_2d(img, size=size)


def _median_filter_2d(img: np.ndarray, size: int) -> np.ndarray:
    """Apply a square median filter to a 2D array.

    Args:
        img (np.ndarray): 2D image array.
        size (int): Window size.

    Returns:
        np.ndarray: Median-filtered image.
    """
    if img.ndim != 2:
        raise ValueError("_median_filter_2d expects a 2D array")

    window = max(int(size), 1)
    pad = window // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    return np.median(windows, axis=(-2, -1)).astype(img.dtype, copy=False)


def _is_sentinel1_image(image_path: Path) -> bool:
    """Return whether an image path refers to Sentinel-1 imagery.

    Args:
        image_path (Path): Image path to inspect.

    Returns:
        bool: True when the filename stem ends with an `S1` token.

    Example:
        >>> _is_sentinel1_image(Path("2023-05-18-14-08-31_RGB_S1.jpg"))
        True
        >>> _is_sentinel1_image(Path("2023-05-18-14-08-31_S1_RGB.jpg"))
        False
    """
    stem_tokens = [
        token.upper() for token in re.split(r"[^A-Za-z0-9]+", image_path.stem) if token
    ]
    return bool(stem_tokens) and stem_tokens[-1] == "S1"


def _emit_progress(
    current: int,
    total: int,
    written: int,
    skipped: int,
    failures: int,
    image: str = "",
) -> None:
    """Emit structured progress events for parent-process consumers."""
    payload = {
        "current": current,
        "total": total,
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "image": image,
    }
    print(PROGRESS_PREFIX + json.dumps(payload), file=sys.stderr, flush=True)


def _update_progress(
    progress_bar: tqdm,
    current: int,
    total: int,
    written: int,
    skipped: int,
    failures: int,
    image: str = "",
) -> None:
    """Sync terminal tqdm state and structured parent progress updates."""
    progress_bar.set_postfix(
        written=written,
        skipped=skipped,
        failures=failures,
    )
    _emit_progress(current, total, written, skipped, failures, image)


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for standalone image segmentation."""

    input_dir: Path
    output_dir: Path
    model_name_or_path: str
    implementation: str = "BEST"
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    overwrite: bool = False
    use_gpu: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """Container for one loaded model and metadata."""

    model: Any
    model_type: str
    target_size: tuple[int, int]
    nclasses: int
    class_mapping: dict[int, str]


@dataclass(frozen=True)
class InferenceBackend:
    """Container for Keras .h5 segmentation backend."""

    models: tuple[ModelSpec, ...]


@dataclass(frozen=True)
class PreparedImage:
    """Preprocessed image payload for segmentation inference."""

    standardized_image: np.ndarray
    source_shape: tuple[int, int]


def filter_no_data_pixels(files: list[str], percent_no_data: float = 0.50) -> list[str]:
    """
    Filters out image files that have a percentage of black (no data) pixels greater than the specified threshold.
    Args:
        files (list[str]): A list of file paths to image files.
        percent_no_data (float, optional): The maximum allowed percentage of black pixels in an image.
                                           Images with a higher percentage of black pixels will be filtered out.
                                           Defaults to 0.50 (50%).
    Returns:
        list[str]: A list of file paths to images that have a percentage of black pixels less than or equal to the specified threshold.
    """

    def percentage_of_black_pixels(img: Image.Image) -> float:
        # Calculate the total number of pixels in the image
        num_total_pixels = img.size[0] * img.size[1]
        img_array = np.array(img)
        # Count the number of black pixels in the image
        black_pixels = np.count_nonzero(np.all(img_array == 0, axis=-1))
        # Calculate the percentage of black pixels
        percentage = black_pixels / num_total_pixels
        return percentage

    valid_images = []

    def has_image_files(file_list, extensions):
        return any(file.lower().endswith(extensions) for file in file_list)

    extensions = (".jpg", ".png", ".jpeg")
    contains_images = has_image_files(files, extensions)
    if not contains_images:
        return files

    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img = Image.open(file)
            percentage = percentage_of_black_pixels(img)
            if percentage <= percent_no_data:
                valid_images.append(file)

    return valid_images


def create_metadatadict(
    nclasses: int,
    n_data_bands: int,
    input_file: str,
    est_label: np.ndarray,
    class_mapping: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Create prediction metadata for NPZ serialization.

    Args:
        nclasses (int): Number of classes.
        n_data_bands (int): Number of image bands.
        input_file (str): Source image path.
        est_label (np.ndarray): Predicted label array.
        class_mapping (dict[int, str] | None): Optional class index-to-name map.

    Returns:
        dict[str, Any]: Metadata dictionary used for compressed NPZ output.
    """
    resolved_class_mapping = class_mapping or {
        class_index: f"class_{class_index}" for class_index in range(nclasses)
    }
    return {
        "nclasses": nclasses,
        "n_data_bands": n_data_bands,
        "input_file": input_file,
        "grey_label": est_label.astype(np.uint8),
        "av_softmax_scores": est_label,
        "classes_to_index": resolved_class_mapping,
    }


def create_npz(npz_path: str, metadata_dict: dict[str, Any]) -> None:
    """Save metadata to a compressed NPZ file.

    Args:
        npz_path (str): Output NPZ path.
        metadata_dict (dict[str, Any]): Metadata payload to store.
    """
    np.savez_compressed(npz_path, **metadata_dict)


def infer_npz_output_path(predseg_output_path: Path) -> Path:
    """Infer CoastSeg-style NPZ path from a predseg PNG path.

    Args:
        predseg_output_path (Path): Path to a `_predseg.png` output.

    Returns:
        Path: Companion `_res.npz` path.
    """
    return predseg_output_path.with_name(
        predseg_output_path.name.replace("_predseg.png", "_res.npz")
    )


def build_prediction_metadata(
    mask: np.ndarray,
    input_file: str,
    n_data_bands: int,
    class_mapping: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Build metadata payload for a prediction artifact.

    Args:
        mask (np.ndarray): Predicted class mask.
        input_file (str): Source image path.
        n_data_bands (int): Number of image bands.
        class_mapping (dict[int, str] | None): Optional class index-to-name map.

    Returns:
        dict[str, Any]: Metadata dictionary for NPZ serialization.
    """
    nclasses = max(int(mask.max()) + 1, 1)
    return create_metadatadict(
        nclasses=nclasses,
        n_data_bands=n_data_bands,
        input_file=input_file,
        est_label=mask,
        class_mapping=class_mapping,
    )


def save_prediction_npz(
    predseg_output_path: Path,
    mask: np.ndarray,
    input_file: str,
    n_data_bands: int,
    class_mapping: dict[int, str] | None = None,
) -> Path:
    """Save one CoastSeg-style `_res.npz` artifact.

    Args:
        predseg_output_path (Path): Path to the corresponding `_predseg.png` output.
        mask (np.ndarray): Predicted class mask.
        input_file (str): Source image path.
        n_data_bands (int): Number of image bands.
        class_mapping (dict[int, str] | None): Optional class index-to-name map.

    Returns:
        Path: Saved NPZ output path.
    """
    npz_output_path = infer_npz_output_path(predseg_output_path)
    metadata_dict = build_prediction_metadata(
        mask=mask,
        input_file=input_file,
        n_data_bands=n_data_bands,
        class_mapping=class_mapping,
    )
    create_npz(str(npz_output_path), metadata_dict)
    return npz_output_path


def _set_tf_device_mode(use_gpu: bool) -> None:
    """Configure TensorFlow GPU visibility.

    Args:
        use_gpu (bool): If False, force TensorFlow to CPU mode.
    """
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass


def collect_image_paths(input_dir: Path, image_extensions: Sequence[str]) -> list[Path]:
    """Collect supported image paths recursively.

    Args:
        input_dir (Path): Root directory to scan.
        image_extensions (Sequence[str]): Allowed filename extensions.

    Returns:
        list[Path]: Sorted list of matching image paths.
    """
    suffixes = {ext.lower() for ext in image_extensions}
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )


def _is_model_directory(model_name_or_path: str) -> bool:
    """Check whether a path looks like a model directory.

    Args:
        model_name_or_path (str): Candidate model directory path.

    Returns:
        bool: True if directory exists and has both `.h5` and `.json` files.
    """
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        return False
    has_h5 = any(model_path.glob("*.h5"))
    has_json = any(model_path.glob("*.json"))
    return has_h5 and has_json


def _resolve_weight_files(model_dir: Path, implementation: str) -> list[Path]:
    """Resolve model weight files from model directory.

    Args:
        model_dir (Path): Directory containing model files.
        implementation (str): Either BEST or ENSEMBLE.

    Returns:
        list[Path]: Ordered list of weight files.

    Raises:
        FileNotFoundError: If expected model files are missing.
        ValueError: If implementation is invalid.
    """
    mode = implementation.upper()
    if mode == "BEST":
        best_model_file = model_dir / "BEST_MODEL.txt"
        if not best_model_file.exists():
            raise FileNotFoundError(
                f"BEST_MODEL.txt not found in model dir: {model_dir}"
            )
        model_name = best_model_file.read_text(encoding="utf-8").strip()
        weights_path = model_dir / model_name
        if not weights_path.exists():
            raise FileNotFoundError(f"BEST model weights not found: {weights_path}")
        return [weights_path]

    if mode == "ENSEMBLE":
        weight_files = sorted(model_dir.glob("*.h5"))
        if not weight_files:
            raise FileNotFoundError(f"No .h5 files found in model dir: {model_dir}")
        return list(weight_files)

    raise ValueError(
        f"Invalid implementation '{implementation}'. Use BEST or ENSEMBLE."
    )


def _load_model_config(weights_path: Path) -> dict[str, Any]:
    """Load model config JSON for one weights file.

    Args:
        weights_path (Path): Path to model weights.

    Returns:
        dict[str, Any]: Parsed model config dictionary.

    Raises:
        FileNotFoundError: If the matching config JSON is missing.
    """
    config_file = weights_path.with_suffix(".json")
    config_file = Path(str(config_file).replace("weights", "config"))
    if "_fullmodel" in config_file.stem:
        config_file = config_file.with_name(config_file.name.replace("_fullmodel", ""))
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return json.loads(config_file.read_text(encoding="utf-8"))


def _load_modelcard_data(model_dir: Path) -> dict[str, Any] | None:
    """Load modelcard JSON when present.

    Args:
        model_dir (Path): Model directory should should contain a `*modelcard.json` file.

    Returns:
        dict[str, Any] | None: Parsed modelcard data, or None if unavailable.
    """
    modelcard_files = sorted(model_dir.glob("*modelcard.json"))
    if not modelcard_files:
        return None

    try:
        return json.loads(modelcard_files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_model_info_style_classes(
    modelcard_data: dict[str, Any],
) -> dict[int, str] | None:
    """Extract class mapping using ModelInfo-style keys.

    Args:
        modelcard_data (dict[str, Any]): Parsed modelcard dictionary.

    Returns:
        dict[int, str] | None: Class mapping if found, else None.
    """
    for section in ("DATASET", "DATASET1"):
        section_data = modelcard_data.get(section)
        if isinstance(section_data, dict) and isinstance(
            section_data.get("CLASSES"), dict
        ):
            return {
                int(key): str(value) for key, value in section_data["CLASSES"].items()
            }

    top_level_classes = modelcard_data.get("CLASSES")
    if isinstance(top_level_classes, dict):
        return {int(key): str(value) for key, value in top_level_classes.items()}

    return None


def _resolve_class_mapping(
    config: dict[str, Any],
    nclasses: int,
    model_dir: Path,
) -> dict[int, str]:
    """Resolve class index-to-name mapping.

    Args:
        config (dict[str, Any]): Model config dictionary.
            Expected to contain keys like CLASS_MAPPING, CLASSES, or ID2LABEL for explicit mapping.
        nclasses (int): Number of classes the model predicts, used to build default mapping.
            This is the number of classes the model was trained to predict. It cannot be changed at inference time.
        model_dir (Path): Model directory that may contain a modelcard.

    Returns:
        dict[int, str]: Final class mapping for all class indices.
    """
    default_model_info_mapping = {
        0: "water",
        1: "whitewater",
        2: "sediment",
        3: "other",
    }
    default_mapping = {
        class_index: default_model_info_mapping.get(class_index, f"class_{class_index}")
        for class_index in range(nclasses)
    }

    modelcard_data = _load_modelcard_data(model_dir)
    if isinstance(modelcard_data, dict):
        extracted_classes = _extract_model_info_style_classes(modelcard_data)
        if extracted_classes:
            return {
                **default_mapping,
                **extracted_classes,
            }  # merge with defaults, giving precedence to extracted mapping from modelcard

    explicit_mapping = config.get("CLASS_MAPPING")
    # map each class index to a string name, ignoring any entries that cannot be parsed as int->str
    if isinstance(explicit_mapping, dict):
        resolved: dict[int, str] = {}
        for class_index, class_name in explicit_mapping.items():
            try:
                resolved[int(class_index)] = str(class_name)
            except (TypeError, ValueError):
                continue
        if resolved:
            return {
                **default_mapping,
                **resolved,
            }  # merge the dictionaries, with explicit mapping taking precedence over defaults

    labels_like = (
        config.get("CLASSES") or config.get("CLASS_LABELS") or config.get("LABELS")
    )
    if isinstance(labels_like, list) and labels_like:
        return {
            class_index: (
                str(labels_like[class_index])
                if class_index < len(labels_like)
                else default_mapping[class_index]
            )
            for class_index in range(nclasses)
        }

    id2label = config.get("id2label") or config.get("ID2LABEL")
    if isinstance(id2label, dict):
        resolved = {}
        for class_index, class_name in id2label.items():
            try:
                resolved[int(class_index)] = str(class_name)
            except (TypeError, ValueError):
                continue
        if resolved:
            return {**default_mapping, **resolved}

    return default_mapping


def _build_custom_model(config: dict[str, Any]) -> Any:
    """Build a SegFormer model when direct load fails.

    Args:
        config (dict[str, Any]): Model config JSON.

    Returns:
        Any: Uninitialized Keras model object.

    Raises:
        ImportError: If required optional dependency is missing.
        ValueError: If model type is not segformer.
    """
    try:
        from doodleverse_utils.model_imports import (  # type: ignore
            segformer,
        )
    except ImportError as exc:
        raise ImportError(
            "SegFormer models require doodleverse_utils. "
            "Install with: pip install doodleverse-utils"
        ) from exc

    target_size = tuple(config.get("TARGET_SIZE", [512, 512]))
    model_type = str(config.get("MODEL", "")).lower()
    nclasses = int(config.get("NCLASSES", 2))
    n_data_bands = int(config.get("N_DATA_BANDS", 3))
    if model_type != "segformer":
        raise ValueError(
            f"Unsupported model type '{model_type}'. This script supports only segformer."
        )

    # Ensure config provides expected image shape metadata for segformer init.
    _ = (target_size, n_data_bands)
    id2label = {label_id: str(label_id) for label_id in range(nclasses)}
    model = segformer(id2label, num_classes=nclasses)

    # Mirror zoo_model behavior by compiling before loading weights.
    model_obj = model[0] if isinstance(model, tuple) else model
    # model_obj.compile(optimizer="adam", loss=dice_coef_loss(nclasses)) # original code
    model_obj.compile(optimizer="adam")

    return model


def _load_one_model(weights_path: Path) -> ModelSpec:
    """Load one model and its metadata.

    Args:
        weights_path (Path): Path to model weights.

    Returns:
        ModelSpec: Loaded model plus resolved metadata.

    Raises:
        ValueError: If model type is unsupported.
    """
    config = _load_model_config(weights_path)
    model_type = str(config.get("MODEL", "")).lower()
    if model_type != "segformer":
        raise ValueError(
            f"Unsupported model type '{model_type}' in {weights_path}. Only segformer is supported."
        )
    target_size_raw = config.get("TARGET_SIZE", [512, 512])
    target_size = (int(target_size_raw[0]), int(target_size_raw[1]))
    nclasses = int(config.get("NCLASSES", 2))
    class_mapping = _resolve_class_mapping(config, nclasses, weights_path.parent)

    try:
        model = tf.keras.models.load_model(weights_path, compile=False)
    except Exception:
        model = _build_custom_model(config)
        model_obj = model[0] if isinstance(model, tuple) else model
        model_obj.load_weights(weights_path)

    return ModelSpec(
        model=model,
        model_type=model_type,
        target_size=target_size,
        nclasses=nclasses,
        class_mapping=class_mapping,
    )


def load_models(model_dir: Path, implementation: str) -> InferenceBackend:
    """Load one or more models for inference.

    Args:
        model_dir (Path): Directory containing model files.
        implementation (str): `BEST` or `ENSEMBLE` selection mode.

    Returns:
        InferenceBackend: Inference backend with loaded models.

    Raises:
        FileNotFoundError: If no loadable models are found.
    """
    weight_files = _resolve_weight_files(model_dir, implementation)
    loaded_models = tuple(_load_one_model(weight_file) for weight_file in weight_files)
    if not loaded_models:
        raise FileNotFoundError(f"No loadable models found in {model_dir}")
    return InferenceBackend(models=loaded_models)


def _standardize(image: np.ndarray) -> np.ndarray:
    """Standardize an image with adjusted standard deviation.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Standardized `float32` image.
    """
    pixel_count = image.shape[0] * image.shape[1]
    std_value = max(float(np.std(image)), 1.0 / np.sqrt(pixel_count))
    mean_value = float(np.mean(image))
    standardized = (image - mean_value) / std_value
    if standardized.ndim == 2:
        standardized = np.dstack((standardized, standardized, standardized))
    return standardized.astype("float32")


def _forward_logits(image_input: np.ndarray, model_spec: ModelSpec) -> np.ndarray:
    """Run one model forward pass and return logits.

    Args:
        image_input (np.ndarray): Model input tensor.
        model_spec (ModelSpec): Loaded model metadata.

    Returns:
        np.ndarray: Logits in `H x W x C` layout.

    Raises:
        RuntimeError: If model inference fails.
        ValueError: If logits shape is unsupported.
    """
    batched = tf.expand_dims(image_input, 0)
    model_obj = (
        model_spec.model[0] if isinstance(model_spec.model, tuple) else model_spec.model
    )

    try:
        logits = model_obj(batched).logits
    except Exception:
        try:
            logits = model_obj.predict(batched, batch_size=1)
            logits = logits.logits if hasattr(logits, "logits") else logits
        except Exception as exc:
            raise RuntimeError("SegFormer forward pass failed") from exc

    logits_np = np.asarray(logits).astype("float32")
    if logits_np.ndim == 4 and logits_np.shape[0] == 1:
        logits_np = logits_np[0]

    if logits_np.ndim == 3 and logits_np.shape[-1] == model_spec.nclasses:
        return logits_np

    if logits_np.ndim == 3 and logits_np.shape[0] == model_spec.nclasses:
        return np.transpose(logits_np, (1, 2, 0))

    raise ValueError(
        f"Unsupported logits shape {logits_np.shape} for model type {model_spec.model_type}"
    )


def _prepare_model_input(
    standardized_hwc_image: np.ndarray,
    model_spec: ModelSpec,
) -> np.ndarray:
    """Prepare standardized image tensor layout for one model.

    Args:
        standardized_hwc_image (np.ndarray): Standardized image in `H x W x C` layout.
        model_spec (ModelSpec): Loaded model metadata.

    Returns:
        np.ndarray: Input tensor in `C x H x W` layout.
    """
    _ = model_spec
    return np.transpose(standardized_hwc_image, (2, 0, 1))


def preprocess_image(
    image_rgb: np.ndarray,
    target_size: tuple[int, int],
    apply_smoothing: bool = False,
) -> PreparedImage:
    """Resize and standardize an image for inference.

    Args:
        image_rgb (np.ndarray): Input RGB image.
        target_size (tuple[int, int]): Model target size as `(height, width)`.
        apply_smoothing (bool): Whether to median smooth before resizing.

    Returns:
        PreparedImage: Standardized resized image plus original spatial shape.
    """
    target_height, target_width = target_size
    image_to_resize = median_smooth(image_rgb) if apply_smoothing else image_rgb
    resized_image = np.asarray(
        Image.fromarray(image_to_resize).resize(
            (target_width, target_height), resample=Image.Resampling.BILINEAR
        )
    )
    standardized_image = _standardize(resized_image)
    return PreparedImage(
        standardized_image=standardized_image,
        source_shape=tuple(image_rgb.shape[:2]),
    )


def predict_mask(
    prepared_image: PreparedImage,
    backend: InferenceBackend,
) -> np.ndarray:
    """Predict a segmentation mask from preprocessed model input.

    Args:
        prepared_image (PreparedImage): Standardized model input plus source shape.
        backend (InferenceBackend): Loaded inference backend.

    Returns:
        np.ndarray: Predicted class mask in source image size.

    Raises:
        ValueError: If ensemble models have incompatible metadata.
    """
    first_model = backend.models[0]
    model_logits: list[np.ndarray] = []
    for model_spec in backend.models:
        if model_spec.target_size != first_model.target_size:
            raise ValueError("All ensemble models must share the same TARGET_SIZE")
        if model_spec.nclasses != first_model.nclasses:
            raise ValueError("All ensemble models must share the same NCLASSES")
        model_input = _prepare_model_input(
            prepared_image.standardized_image,
            model_spec,
        )
        # Run the model forward pass to get logits and collect them for averaging.
        model_logits.append(_forward_logits(model_input, model_spec))

    avg_logits = np.mean(np.stack(model_logits, axis=0), axis=0)
    resized_logits = tf.image.resize(
        avg_logits, size=prepared_image.source_shape, method="bilinear"
    )
    return tf.argmax(resized_logits, axis=-1).numpy()


def load_inference_backend(
    model_name_or_path: str,
    implementation: str,
) -> InferenceBackend:
    """Load inference backend from a model directory.

    Args:
        model_name_or_path (str): Model directory path.
        implementation (str): `BEST` or `ENSEMBLE` selection mode.

    Returns:
        InferenceBackend: Loaded inference backend.

    Raises:
        ValueError: If the model path is not a valid model directory.
    """
    if not _is_model_directory(model_name_or_path):
        raise ValueError(
            "--model must point to a model directory containing .h5 and .json files"
        )
    return load_models(Path(model_name_or_path), implementation)


def _label_to_colors(
    labels: np.ndarray,
    nodata_mask: np.ndarray,
    colormap: list[str],
) -> np.ndarray:
    """Map label indices to RGB colors.
    No data pixels are set to black regardless of the colormap.

    Args:
        labels (np.ndarray): Label image with class indices.
        nodata_mask (np.ndarray): Boolean mask for no-data pixels.
        colormap (list[str]): Color palette as hex strings.

    Returns:
        np.ndarray: Colorized RGB label image.
    """

    def _from_hex(hex_value: str) -> tuple[int, int, int]:
        cleaned = hex_value.lstrip("#")
        return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    palette = [_from_hex(color) for color in colormap]
    color_image = np.zeros(labels.shape[:2] + (3,), dtype=np.uint8)

    min_label = int(np.min(labels))
    max_label = int(np.max(labels))
    for label_id in range(min_label, max_label + 1):
        color_image[labels == label_id] = palette[label_id % len(palette)]

    color_image[nodata_mask] = (0, 0, 0)
    return color_image


def save_mask(mask: np.ndarray, image_rgb: np.ndarray, output_path: Path) -> None:
    """Save a colorized prediction mask as `_predseg.png`.

    Args:
        mask (np.ndarray): Predicted class mask.
        image_rgb (np.ndarray): Source RGB image used to compute no-data mask.
        output_path (Path): Output PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_int = mask.astype(np.uint8)

    class_label_colormap = [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
        "#316395",
        "#ffe4e1",
        "#ff7373",
        "#666666",
        "#c0c0c0",
        "#66cdaa",
        "#afeeee",
        "#0e2f44",
        "#420420",
        "#794044",
        "#3399ff",
    ]

    nclasses = max(int(mask_int.max()) + 1, 1)
    color_map = class_label_colormap[:nclasses]
    nodata_mask = image_rgb[:, :, 0] == 0
    color_label = _label_to_colors(mask_int, nodata_mask, color_map)
    Image.fromarray(color_label.astype(np.uint8), mode="RGB").save(output_path)


def save_prediction_artifacts(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    predseg_output_path: Path,
    input_file: str,
    class_mapping: dict[int, str] | None = None,
) -> Path:
    """Save prediction PNG outputs plus a companion CoastSeg-style NPZ file.

    Args:
        mask (np.ndarray): HxW array of predicted class indices.
        image_rgb (np.ndarray): HxWx3 RGB image array used for prediction.
        predseg_output_path (Path): Path where the colorized predseg PNG will be saved
        input_file (str): Original input file path to include in NPZ metadata.
        class_mapping (dict[int, str] | None): Optional class index-to-name mapping for NPZ metadata.
    Returns:
        Path: Path to the saved NPZ file.
    """
    save_mask(mask, image_rgb, predseg_output_path)
    return save_prediction_npz(
        predseg_output_path=predseg_output_path,
        mask=mask,
        input_file=input_file,
        n_data_bands=int(image_rgb.shape[-1]) if image_rgb.ndim == 3 else 1,
        class_mapping=class_mapping,
    )


def infer_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    """Build output path while preserving relative structure.

    Args:
        input_path (Path): Source image path.
        input_root (Path): Root input directory.
        output_root (Path): Root output directory.

    Returns:
        Path: Output path ending in `_predseg.png`.
    """
    # Preserve relative path and filename, but change suffix to _predseg.png
    relative = input_path.relative_to(input_root)
    return output_root / relative.with_name(f"{relative.stem}_predseg.png")


def _write_summary(summary: dict[str, object], output_dir: Path) -> None:
    """Write the workflow summary JSON into the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "segmentation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_segmentation_workflow(config: SegmentationConfig) -> dict[str, object]:
    """Run end-to-end segmentation over all images in input_dir.

    Args:
        config (SegmentationConfig): Runtime segmentation configuration.

    Returns:
        dict[str, object]: Summary with counts, paths, and failures.

    Raises:
        FileNotFoundError: If input_dir does not exist.
        ValueError: If no supported images are found.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    failures: list[dict[str, str]] = []
    written = 0
    npz_written = 0
    skipped = 0
    workflow_error: str | None = None

    try:
        if not config.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {config.input_dir}"
            )
        # Recursively collect image paths with supported extensions, then filter out images that are mostly no-data pixels.
        image_paths = collect_image_paths(config.input_dir, config.image_extensions)
        if not image_paths:
            raise ValueError(
                f"No images found in {config.input_dir} with extensions {config.image_extensions}"
            )

        image_paths = [
            Path(p)
            for p in filter_no_data_pixels(
                [str(p) for p in image_paths], percent_no_data=0.50
            )
        ]

        # load the inference backend once before processing images
        backend = load_inference_backend(
            config.model_name_or_path,
            config.implementation,
        )

        progress_bar = tqdm(
            image_paths,
            desc="Running predictions",
            unit="image",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )
        total_images = len(image_paths)
        _emit_progress(0, total_images, written, skipped, len(failures))
        for index, image_path in enumerate(progress_bar, start=1):
            # Infer output path and skip if it exists and overwrite is False
            output_path = infer_output_path(
                image_path, config.input_dir, config.output_dir
            )
            if output_path.exists() and not config.overwrite:
                skipped += 1
                _update_progress(
                    progress_bar,
                    index,
                    total_images,
                    written,
                    skipped,
                    len(failures),
                    image=image_path.name,
                )
                continue

            try:
                image_rgb = np.array(Image.open(image_path).convert("RGB"))
                model_input = preprocess_image(
                    image_rgb,
                    backend.models[0].target_size,
                    apply_smoothing=_is_sentinel1_image(image_path),
                )
                # Run the model(s) to predict the segmentation mask for this image
                mask = predict_mask(
                    model_input,
                    backend=backend,
                )
                # Save prediction artifacts including the segmented PNG and NPZ with metadata.
                save_prediction_artifacts(
                    mask=mask,
                    image_rgb=image_rgb,
                    predseg_output_path=output_path,
                    input_file=str(image_path),
                    class_mapping=backend.models[0].class_mapping,
                )
                written += 1
                npz_written += 1
            except Exception as exc:
                failures.append({"image": str(image_path), "error": str(exc)})

            _update_progress(
                progress_bar,
                index,
                total_images,
                written,
                skipped,
                len(failures),
                image=image_path.name,
            )
    except Exception as exc:
        workflow_error = str(exc)
        raise
    finally:
        summary: dict[str, object] = {
            "config": {
                **asdict(config),
                "input_dir": str(config.input_dir),
                "output_dir": str(config.output_dir),
            },
            "total_images": len(image_paths),
            "written_masks": written,
            "written_npz": npz_written,
            "skipped_existing": skipped,
            "failures": failures,
        }
        if workflow_error is not None:
            summary["error"] = workflow_error
        _write_summary(summary, config.output_dir)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for standalone segmentation.

    Returns:
        argparse.ArgumentParser: Configured parser with workflow arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone image segmentation without coastseg dependencies. "
            "Supports Keras .h5 model directories."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where prediction masks and summary JSON are saved.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model directory containing .h5 weights and .json config files needed for inference.",
    )
    parser.add_argument(
        "--implementation",
        default="BEST",
        choices=["BEST", "ENSEMBLE", "best", "ensemble"],
        help=(
            "Model selection mode. BEST uses BEST_MODEL.txt; "
            "ENSEMBLE loads all .h5 files in model dir."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force TensorFlow to run on CPU.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Parse CLI arguments and execute segmentation workflow.

    Args:
        argv (Iterable[str] | None): Optional CLI args for testing.

    Returns:
        int: Exit code where 0 means success.
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    _set_tf_device_mode(not args.cpu_only)

    config = SegmentationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model,
        implementation=str(args.implementation).upper(),
        overwrite=args.overwrite,
        use_gpu=not args.cpu_only,
    )

    summary = run_segmentation_workflow(config)
    failures = summary.get("failures", [])
    if isinstance(failures, list) and failures:
        first_failure = failures[0] if isinstance(failures[0], dict) else {}
        failed_image = first_failure.get("image", "<unknown image>")
        failure_message = first_failure.get("error", "<unknown error>")
        print(
            FAILURE_PREFIX + "Segmentation completed with image failures: "
            f"{len(failures)} failure(s). "
            f"First failure: {failed_image}: {failure_message}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
