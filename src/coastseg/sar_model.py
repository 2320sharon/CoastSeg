"""Sentinel-1 SAR land/water segmentation with an exported ONNX model.

The model takes three real SAR channels -- ``VV``, ``VH`` and ``VV-VH``, all in dB --
and returns a per-pixel water probability. Every preprocessing constant
(channel order, normalization statistics, output stride, nodata code, class names) is
read from the ``.onnx`` file's own ``metadata_props`` at load time, so swapping in a
retrained model cannot silently desynchronize the constants.

This runs in-process in the main CoastSeg environment: ``onnxruntime`` and ``gdal`` are
both already dependencies here, and neither exists in the ``segmentation_workflow``
TensorFlow subprocess environment.

The batch driver writes the same artifacts the TensorFlow zoo path writes
(``*_predseg.png``, ``*_res.npz``, ``model_settings.json``, ``model_info.json``,
``segmentation_summary.json``) so the existing npz-driven shoreline extraction consumes
SAR output with no changes.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from coastsat import SDS_sar_model, SDS_tools
from osgeo import gdal, osr
from PIL import Image

from coastseg import core_utilities

gdal.UseExceptions()

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SAR_MODEL_DIRNAME",
    "SarModelError",
    "MissingPolarizationError",
    "UnsupportedSpeckleFilterError",
    "SarModelSpec",
    "SarPrediction",
    "describe_sar_polarization_remedy",
    "find_sar_onnx_file",
    "find_local_sar_model",
    "is_sar_model_directory",
    "load_sar_model_spec",
    "segment_scene",
    "sar_nodata_mask",
    "run_sar_segmentation",
]

# Directory under <CoastSeg>/models where a user drops a model by hand. Nothing ships
# there: the default model is downloaded on first use (see find_sar_onnx_file), so this
# directory is routinely absent or empty.
DEFAULT_SAR_MODEL_DIRNAME = "SAR_segmentation_model"

# Speckle filters this module knows how to apply. The model records which filter its
# training data used; anything other than "none" would have to be implemented at step 5
# of the preprocessing pipeline before this module can honour it.
SUPPORTED_SPECKLE_FILTERS = frozenset({"none", ""})

# Guard against accidentally feeding a whole-country mosaic to a fully convolutional
# model. A 3000x3000 scene peaks around 220 MB, so this leaves generous headroom while
# still failing with a readable message instead of an OOM kill.
MAX_INPUT_PIXELS = 80_000_000

# Same palette the TensorFlow zoo path uses, so the good/bad segmentation classifier
# sees the colours it was trained on.
CLASS_LABEL_COLORMAP = (
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
)

_POLARIZATION_TOKEN_RE = re.compile(
    r"_(%s)(?=(?:_dup\d+)?\.tif$)" % "|".join(SDS_tools.SAR_POLARIZATIONS),
    re.IGNORECASE,
)
_DUP_TOKEN_RE = re.compile(r"_(dup\d+)\.tif$", re.IGNORECASE)

# Cache sessions by (onnx path, providers) so a batch run loads the model once.
_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], "ort.InferenceSession"] = {}


class SarModelError(RuntimeError):
    """Base class for every failure raised by the SAR segmentation path."""


class MissingPolarizationError(SarModelError):
    """Raised when a scene or a site lacks a polarization the model requires."""


class UnsupportedSpeckleFilterError(SarModelError):
    """Raised when the model was trained with a speckle filter this module cannot apply."""


def describe_sar_polarization_remedy() -> str:
    """Return the advice to give a user whose site is missing a requested polarization.

    coastsat treats an S1 date as downloaded only when *every* requested polarization is
    present, so re-running the download fetches the missing band and overwrites the
    existing scene in place rather than skipping it as already downloaded. Route any new
    message about missing polarizations through this rather than restating it.
    """
    return (
        "Re-run the download for this site with the polarizations you want; the "
        "missing ones will be fetched and the existing scenes overwritten in place."
    )


# ---------------------------------------------------------------------------
# coastsat interop
# ---------------------------------------------------------------------------


def get_sar_polarizations() -> Tuple[str, ...]:
    """Return the canonical polarization order, as coastsat defines it."""
    return tuple(SDS_tools.SAR_POLARIZATIONS)


def get_polarization_from_filename(filename: str) -> Optional[str]:
    """Return the polarization token in ``filename``, or None when there isn't one.

    Delegates to coastsat's implementation so the two cannot drift.

    Args:
        filename (str): A Sentinel-1 GeoTIFF path or basename.

    Returns:
        Optional[str]: ``"VV"``, ``"VH"``, ``"HH"``, ``"HV"`` or None.
    """
    return SDS_tools.get_polarization_from_filename(filename)


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SarModelSpec:
    """Everything needed to preprocess for, run, and postprocess this ONNX model.

    Every field is read from the ``.onnx`` file's ``metadata_props``; nothing here is
    hardcoded against a particular checkpoint.
    """

    onnx_path: str
    input_name: str
    prob_output: str
    channel_order: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    output_stride: int
    nodata: int
    classes: Dict[int, str]
    input_scale: str = "dB"
    speckle_filter: Dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    water_threshold: float = 0.5

    @property
    def water_class_indices(self) -> List[int]:
        """Class indices whose label names mean water."""
        water_names = {"water", "whitewater"}
        return sorted(
            index
            for index, name in self.classes.items()
            if str(name).strip().lower() in water_names
        )

    @property
    def required_polarizations(self) -> Tuple[str, ...]:
        """Polarizations that must be on disk to build every model channel.

        Derived from ``channel_order``, so a difference channel such as ``VV-VH``
        contributes both of its operands.
        """
        polarizations: List[str] = []
        for channel in self.channel_order:
            for token in str(channel).upper().split("-"):
                token = token.strip()
                if token and token not in polarizations:
                    polarizations.append(token)
        canonical = get_sar_polarizations()
        ordered = [p for p in canonical if p in polarizations]
        ordered += [p for p in polarizations if p not in ordered]
        return tuple(ordered)


def find_sar_onnx_file(model_directory: str, download: bool = True) -> str:
    """Return the ``.onnx`` file to run, fetching the default model if there isn't one.

    The model is not shipped with CoastSeg -- it is ~130 MB, and ``models/**/*.onnx`` is
    gitignored, so a fresh clone has none. It is downloaded from Hugging Face on first
    use instead.

    A copy inside ``model_directory`` always wins, so a custom or retrained model, and
    an offline machine that already has one, keep working untouched. Otherwise this
    defers to ``coastsat.SDS_sar_model``, which pins the download by sha256 and caches
    it via pooch. Reusing coastsat's cache rather than downloading into
    ``model_directory`` is deliberate: both SAR workflows then share one ~130 MB copy
    instead of keeping one each.

    Args:
        model_directory (str): Directory that may hold an exported model.
        download (bool): Fetch the default model when the directory holds none.

    Returns:
        str: Absolute path to the ``.onnx`` file.

    Raises:
        FileNotFoundError: If there is no local model and ``download`` is False.
        SarModelError: If the default model is needed but cannot be obtained.
    """
    directory = Path(model_directory).expanduser() if model_directory else None

    if directory is not None and directory.is_dir():
        candidates = sorted(directory.glob("*.onnx"))
        if len(candidates) > 1:
            logger.warning(
                "Multiple .onnx files in %s; using %s", directory, candidates[0].name
            )
        if candidates:
            return str(candidates[0].resolve())

    if not download:
        raise FileNotFoundError(
            f"No .onnx file found in the SAR model directory: {directory}"
        )

    # already downloaded by either workflow on a previous run
    cached = SDS_sar_model.get_sar_model_path()
    if os.path.isfile(cached):
        return os.path.abspath(cached)

    logger.info(
        "No SAR model in %s; downloading the default model from Hugging Face.",
        directory,
    )
    print(
        "The Sentinel-1 segmentation model (~130 MB) is not on this machine yet and is "
        "being downloaded. This happens once."
    )
    try:
        return SDS_sar_model.fetch_default_sar_model()
    except SDS_sar_model.SarModelUnavailable as error:
        raise SarModelError(
            f"The Sentinel-1 segmentation model was not found in {directory} and could "
            f"not be downloaded: {error}. Download "
            f"{SDS_sar_model.SAR_MODEL_CONFIG['model_name']} manually from "
            f"{SDS_sar_model.SAR_MODEL_CONFIG['url'].split('?')[0]} and place it in "
            f"models/{DEFAULT_SAR_MODEL_DIRNAME}."
        ) from error


def get_sar_model_directory() -> str:
    """Return CoastSeg's designated SAR model directory.

    This is the one place a user can drop a manually downloaded model and have *both*
    SAR workflows pick it up: the zoo path reads it directly, and the coastsat path is
    pointed at it by `find_local_sar_model`.
    """
    return str(
        Path(core_utilities.get_base_dir()) / "models" / DEFAULT_SAR_MODEL_DIRNAME
    )


def find_local_sar_model() -> str:
    """Return a model already sitting in CoastSeg's model directory, or ``""``.

    Never downloads and never raises -- ``""`` simply means "nothing placed by hand",
    which leaves the caller free to fall back to coastsat's own resolution.

    Returns:
        str: Absolute path to the ``.onnx``, or ``""`` when there is none.
    """
    try:
        return find_sar_onnx_file(get_sar_model_directory(), download=False)
    except FileNotFoundError:
        return ""


def is_sar_model_directory(model_directory: Optional[str]) -> bool:
    """Return True when ``model_directory`` is CoastSeg's ONNX SAR model directory.

    True either because it already holds an ``.onnx``, or because it *is* the designated
    SAR model directory, which may legitimately be empty or absent: the model is
    downloaded on first use (see `find_sar_onnx_file`). This routes the run, so it
    has to give the same answer before and after that download -- otherwise the first
    SAR run on a fresh clone would be sent down the TensorFlow path.
    """
    if not model_directory:
        return False
    try:
        directory = Path(model_directory).expanduser()
    except (OSError, ValueError):
        return False
    if directory.name == DEFAULT_SAR_MODEL_DIRNAME:
        return True
    return directory.is_dir() and any(directory.glob("*.onnx"))


def _resolve_providers(use_gpu: bool) -> Tuple[str, ...]:
    """Return the onnxruntime execution providers to try, CPU last."""
    if not use_gpu:
        return ("CPUExecutionProvider",)
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    logger.info("CUDAExecutionProvider is unavailable; running SAR inference on CPU.")
    return ("CPUExecutionProvider",)


def _make_session(onnx_path: str, providers: Tuple[str, ...]) -> "ort.InferenceSession":
    """Create (and cache) an inference session for ``onnx_path``."""
    key = (onnx_path, providers)
    session = _SESSION_CACHE.get(key)
    if session is not None:
        return session
    try:
        session = ort.InferenceSession(onnx_path, providers=list(providers))
    except Exception as error:
        if providers != ("CPUExecutionProvider",):
            logger.warning(
                "Failed to start onnxruntime with %s (%s); falling back to CPU.",
                providers,
                error,
            )
            return _make_session(onnx_path, ("CPUExecutionProvider",))
        raise SarModelError(
            f"Could not load the SAR ONNX model at {onnx_path}: {error}"
        ) from error
    _SESSION_CACHE[key] = session
    return session


def _json_metadata(metadata: Mapping[str, str], key: str, default: Any) -> Any:
    """Read a JSON-encoded metadata value, falling back to ``default``."""
    raw = metadata.get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Could not parse ONNX metadata %r=%r; using default.", key, raw)
        return default


def _resolve_prob_output(
    output_names: Sequence[str], graph_outputs: Sequence[str], onnx_path: str
) -> str:
    """Return the name of the graph output holding ``P(water)``.

    Picking the wrong output does not fail: logits are real numbers too, so thresholding
    them at 0.5 yields a plausible but wrong shoreline. The exported model carries both a
    ``logits`` and a ``water_prob`` output, so guess only when there is nothing to guess
    between.

    Args:
        output_names (Sequence[str]): Output names the model declares in its metadata,
            or the graph's own names when it declares none.
        graph_outputs (Sequence[str]): Output names the ONNX graph actually exposes.
        onnx_path (str): Model path, for error messages.

    Returns:
        str: The output name to request from ``session.run``.

    Raises:
        SarModelError: If the probability output is ambiguous, or if the declared name
            is not one the graph exposes.
    """
    names = [str(name) for name in output_names]
    model_name = os.path.basename(onnx_path)

    if "water_prob" in names:
        chosen = "water_prob"
    elif len(names) == 1:
        # Nothing to disambiguate. Say so rather than silently trusting the name.
        chosen = names[0]
        logger.warning(
            "%s has no output named 'water_prob'; using its only output %r as the water "
            "probability. Values are thresholded directly, so they must be in [0, 1].",
            model_name,
            chosen,
        )
    else:
        raise SarModelError(
            f"{model_name} exposes outputs {names} and none is named 'water_prob', so "
            "which one holds P(water) cannot be determined. Re-export the model with "
            "its probability output named 'water_prob'."
        )

    if graph_outputs and chosen not in graph_outputs:
        raise SarModelError(
            f"{model_name} names {chosen!r} as its probability output, but its graph "
            f"exposes {list(graph_outputs)}."
        )
    return chosen


def load_sar_model_spec(model_directory: str, use_gpu: bool = False) -> SarModelSpec:
    """Read the preprocessing contract out of the model's embedded metadata.

    Args:
        model_directory (str): Directory holding the exported ``.onnx`` model.
        use_gpu (bool): Try the CUDA execution provider before CPU.

    Returns:
        SarModelSpec: The validated model contract.

    Raises:
        FileNotFoundError: If no ``.onnx`` file is present.
        SarModelError: If the model does not carry a usable preprocessing contract.
    """
    onnx_path = find_sar_onnx_file(model_directory)
    session = _make_session(onnx_path, _resolve_providers(use_gpu))
    metadata = session.get_modelmeta().custom_metadata_map or {}

    graph_outputs = [output.name for output in session.get_outputs()]
    output_names = _json_metadata(metadata, "output_names", graph_outputs)
    prob_output = _resolve_prob_output(output_names, graph_outputs, onnx_path)

    graph_input = session.get_inputs()[0]
    input_name = metadata.get("input_name") or graph_input.name

    channel_order = tuple(
        str(channel) for channel in _json_metadata(metadata, "channel_order", [])
    )
    if not channel_order:
        raise SarModelError(
            f"{onnx_path} does not declare 'channel_order' in its ONNX metadata, so "
            "the band stack cannot be built safely. Re-export the model with metadata."
        )

    mean = np.asarray(
        _json_metadata(metadata, "normalization_mean", []), dtype=np.float32
    )
    std = np.asarray(
        _json_metadata(metadata, "normalization_std", []), dtype=np.float32
    )
    if mean.size != len(channel_order) or std.size != len(channel_order):
        raise SarModelError(
            f"{onnx_path}: normalization_mean/std have {mean.size}/{std.size} entries "
            f"but channel_order has {len(channel_order)}."
        )
    if np.any(std == 0):
        raise SarModelError(f"{onnx_path}: normalization_std contains a zero.")

    # Cross-check against the graph itself so a metadata typo cannot go unnoticed.
    graph_channels = graph_input.shape[1] if len(graph_input.shape) == 4 else None
    if isinstance(graph_channels, int) and graph_channels != len(channel_order):
        raise SarModelError(
            f"{onnx_path}: the graph expects {graph_channels} channels but "
            f"channel_order declares {len(channel_order)}."
        )

    raw_classes = _json_metadata(metadata, "classes", {"0": "land", "1": "water"})
    classes = {int(index): str(name) for index, name in dict(raw_classes).items()}

    speckle_filter = _json_metadata(metadata, "speckle_filter", {"type": "none"})
    if not isinstance(speckle_filter, dict):
        speckle_filter = {"type": str(speckle_filter)}

    spec = SarModelSpec(
        onnx_path=onnx_path,
        input_name=str(input_name),
        prob_output=str(prob_output),
        channel_order=channel_order,
        mean=mean,
        std=std,
        output_stride=int(metadata.get("output_stride", 32)),
        nodata=int(metadata.get("nodata", 255)),
        classes=classes,
        input_scale=str(metadata.get("input_scale", "dB")),
        speckle_filter=speckle_filter,
    )
    _check_speckle_filter(spec)
    if not spec.water_class_indices:
        logger.warning(
            "%s declares classes %s, none of which is named 'water'. Shoreline "
            "extraction will not know which class is water.",
            onnx_path,
            spec.classes,
        )
    return spec


def get_session(spec: SarModelSpec, use_gpu: bool = False) -> "ort.InferenceSession":
    """Return the cached inference session for ``spec``."""
    return _make_session(spec.onnx_path, _resolve_providers(use_gpu))


def _check_speckle_filter(spec: SarModelSpec) -> None:
    """Raise when the model needs a speckle filter this module does not implement."""
    filter_type = str(spec.speckle_filter.get("type", "none")).strip().lower()
    if filter_type not in SUPPORTED_SPECKLE_FILTERS:
        raise UnsupportedSpeckleFilterError(
            f"{spec.onnx_path} was trained with the {filter_type!r} speckle filter, "
            "which coastseg.sar_model does not implement. Running without it would "
            "silently degrade accuracy."
        )


# ---------------------------------------------------------------------------
# GDAL raster IO
# ---------------------------------------------------------------------------


def read_band(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...], str]:
    """Read band 1 and its validity mask on the raster's own grid.

    Args:
        path (str): GeoTIFF path.

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[float, ...], str]: ``(values, valid,
        geotransform, projection)`` where ``values`` is float32 ``[H, W]`` and ``valid``
        is a boolean ``[H, W]`` mask.

    Raises:
        SarModelError: If the raster cannot be opened or read.
    """
    dataset = None
    try:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if dataset is None:
            raise SarModelError(f"Could not open raster: {path}")
        band = dataset.GetRasterBand(1)
        values = band.ReadAsArray().astype(np.float32)
        # GetMaskBand folds in the band's nodata value; a raster with no nodata set
        # reports GMF_ALL_VALID, so this degrades safely to "everything is valid".
        mask = band.GetMaskBand().ReadAsArray()
        valid = (mask > 0) & np.isfinite(values)
        geotransform = tuple(dataset.GetGeoTransform())
        projection = dataset.GetProjection()
    except SarModelError:
        raise
    except Exception as error:
        raise SarModelError(f"Failed to read {path}: {error}") from error
    finally:
        dataset = None
    return values, valid, geotransform, projection


def grids_match(
    geotransform_a: Sequence[float],
    projection_a: str,
    shape_a: Tuple[int, int],
    geotransform_b: Sequence[float],
    projection_b: str,
    shape_b: Tuple[int, int],
    atol: float = 1e-6,
) -> bool:
    """Return True when two rasters sit on the same grid (size, transform and CRS)."""
    if tuple(shape_a) != tuple(shape_b):
        return False
    if not np.allclose(
        np.asarray(geotransform_a), np.asarray(geotransform_b), atol=atol
    ):
        return False
    if projection_a == projection_b:
        return True
    if not projection_a or not projection_b:
        return False
    try:
        srs_a = osr.SpatialReference()
        srs_a.ImportFromWkt(projection_a)
        srs_b = osr.SpatialReference()
        srs_b.ImportFromWkt(projection_b)
        return bool(srs_a.IsSame(srs_b))
    except Exception:  # pragma: no cover - malformed WKT
        return False


def _bounds_from_grid(
    geotransform: Sequence[float], shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Return ``(minx, miny, maxx, maxy)`` for a north-up grid."""
    height, width = shape
    origin_x, pixel_width, row_rotation, origin_y, column_rotation, pixel_height = (
        geotransform
    )
    if row_rotation or column_rotation:
        logger.warning(
            "Reference grid is rotated (%s, %s); reprojection bounds are approximate.",
            row_rotation,
            column_rotation,
        )
    far_x = origin_x + width * pixel_width
    far_y = origin_y + height * pixel_height
    return (
        min(origin_x, far_x),
        min(origin_y, far_y),
        max(origin_x, far_x),
        max(origin_y, far_y),
    )


def _mem_dataset(
    array: np.ndarray, geotransform: Sequence[float], projection: str, gdal_type: int
):
    """Wrap a numpy array in an in-memory GDAL dataset carrying ``geotransform``."""
    height, width = array.shape
    dataset = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal_type)
    dataset.SetGeoTransform(tuple(geotransform))
    if projection:
        dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(array)
    return dataset


def read_band_on_grid(
    path: str,
    reference_geotransform: Sequence[float],
    reference_projection: str,
    reference_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Read band 1 aligned onto a reference grid, reprojecting only if it differs.

    VV and VH of one Sentinel-1 GRD acquisition are geocoded together and virtually
    always share a grid, so the fast path is a plain read. When they do differ the data
    is resampled bilinearly and the validity mask nearest-neighbour, matching the
    training pipeline.

    Args:
        path (str): GeoTIFF to read.
        reference_geotransform (Sequence[float]): Target geotransform.
        reference_projection (str): Target projection WKT.
        reference_shape (Tuple[int, int]): Target ``(height, width)``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(values, valid)`` on the reference grid.
    """
    values, valid, geotransform, projection = read_band(path)
    if grids_match(
        geotransform,
        projection,
        values.shape,
        reference_geotransform,
        reference_projection,
        reference_shape,
    ):
        return values, valid

    logger.warning(
        "%s is not on the reference grid (%s vs %s); reprojecting onto it.",
        path,
        values.shape,
        reference_shape,
    )
    height, width = reference_shape
    bounds = _bounds_from_grid(reference_geotransform, reference_shape)
    warp_options = dict(
        format="MEM",
        dstSRS=reference_projection or None,
        outputBounds=bounds,
        width=width,
        height=height,
    )

    source_values = _mem_dataset(values, geotransform, projection, gdal.GDT_Float32)
    warped_values = gdal.Warp(
        "",
        source_values,
        resampleAlg=gdal.GRA_Bilinear,
        dstNodata=float("nan"),
        **warp_options,
    )

    source_mask = _mem_dataset(
        valid.astype(np.uint8), geotransform, projection, gdal.GDT_Byte
    )
    warped_mask = gdal.Warp(
        "",
        source_mask,
        resampleAlg=gdal.GRA_NearestNeighbour,
        dstNodata=0,
        **warp_options,
    )

    aligned_values = warped_values.GetRasterBand(1).ReadAsArray().astype(np.float32)
    aligned_mask = warped_mask.GetRasterBand(1).ReadAsArray()
    aligned_valid = (aligned_mask > 0) & np.isfinite(aligned_values)
    return aligned_values, aligned_valid


# ---------------------------------------------------------------------------
# Preprocessing / inference / postprocessing
# ---------------------------------------------------------------------------


def resolve_scene_polarizations(
    file_paths: Sequence[str], required: Sequence[str]
) -> Dict[str, str]:
    """Map each required polarization to its file path.

    Args:
        file_paths (Sequence[str]): Per-polarization GeoTIFF paths for one scene.
        required (Sequence[str]): Polarizations the model needs.

    Returns:
        Dict[str, str]: ``{polarization: path}``.

    Raises:
        MissingPolarizationError: If any required polarization is absent.
    """
    paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
    found: Dict[str, str] = {}
    for path in paths:
        polarization = get_polarization_from_filename(path)
        if polarization:
            found.setdefault(polarization.upper(), path)

    missing = [polarization for polarization in required if polarization not in found]
    if missing:
        scene = os.path.basename(paths[0]) if paths else "<no files>"
        raise MissingPolarizationError(
            f"{scene}: the SAR model needs {list(required)} but only "
            f"{sorted(found) or 'no polarizations'} were found on disk."
        )
    return {polarization: found[polarization] for polarization in required}


def _derive_channel(name: str, bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return one model channel, computing ``A-B`` differences from raw bands."""
    key = str(name).upper()
    if key in bands:
        return bands[key]
    if "-" in key:
        left, right = key.split("-", 1)
        if left in bands and right in bands:
            return bands[left] - bands[right]
    raise SarModelError(
        f"Cannot build channel {name!r} from the available bands {sorted(bands)}."
    )


def build_sar_stack(
    file_paths: Sequence[str], spec: SarModelSpec
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...], str]:
    """Build the raw dB band stack in the model's channel order.

    Steps 1-7 of the preprocessing contract: read VV on its own grid, read the other
    polarizations aligned to it, intersect the validity masks, and derive difference
    channels *before* any invalid-pixel fill so no fill value leaks into a subtraction.

    Args:
        file_paths (Sequence[str]): Per-polarization GeoTIFF paths for one scene.
        spec (SarModelSpec): The model contract.

    Returns:
        Tuple: ``(stack float32 [C, H, W], valid bool [H, W], geotransform, projection)``
        on the first required polarization's grid.
    """
    _check_speckle_filter(spec)
    required = spec.required_polarizations
    paths = resolve_scene_polarizations(file_paths, required)

    reference_polarization = required[0]
    values, valid, geotransform, projection = read_band(paths[reference_polarization])
    bands: Dict[str, np.ndarray] = {reference_polarization: values}

    for polarization in required[1:]:
        other_values, other_valid = read_band_on_grid(
            paths[polarization], geotransform, projection, valid.shape
        )
        bands[polarization] = other_values
        valid = valid & other_valid

    # A speckle filter would run here, per band, on raw dB. This model uses none.

    stack = np.stack(
        [_derive_channel(channel, bands) for channel in spec.channel_order]
    ).astype(np.float32)
    return stack, valid, geotransform, projection


def preprocess(
    stack: np.ndarray, valid: np.ndarray, spec: SarModelSpec
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Fill, normalize and pad a raw dB stack into the model's input tensor.

    Steps 8-10 of the preprocessing contract. Invalid pixels are filled with the
    per-channel mean so they land at exactly 0 after normalization, and padding is
    applied bottom/right with 0.0 -- neutral, because the image is already normalized.
    The image is never resized: the model was trained at the native ~10 m ground sample
    distance and rescaling measurably degrades shoreline accuracy.

    Args:
        stack (np.ndarray): Raw dB channels ``[C, H, W]``.
        valid (np.ndarray): Boolean validity mask ``[H, W]``.
        spec (SarModelSpec): The model contract.

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: ``(input tensor [1, C, Hp, Wp], (H, W))``.
    """
    height, width = valid.shape
    if height * width > MAX_INPUT_PIXELS:
        raise SarModelError(
            f"Scene is {height}x{width} ({height * width} pixels), above the "
            f"{MAX_INPUT_PIXELS}-pixel limit for whole-scene SAR inference. "
            "Reduce the ROI size and download again."
        )

    image = np.array(stack, dtype=np.float32, copy=True)
    image[:, ~valid] = spec.mean[:, None]
    image = (image - spec.mean[:, None, None]) / spec.std[:, None, None]

    stride = spec.output_stride
    padded_height = -(-height // stride) * stride
    padded_width = -(-width // stride) * stride
    tensor = np.zeros(
        (1, image.shape[0], padded_height, padded_width), dtype=np.float32
    )
    tensor[0, :, :height, :width] = image
    return tensor, (height, width)


def postprocess(
    water_prob: np.ndarray,
    original_shape: Tuple[int, int],
    valid: np.ndarray,
    spec: SarModelSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the padding off, threshold, and stamp nodata back onto invalid pixels.

    Args:
        water_prob (np.ndarray): Model output, ``[1, 1, Hp, Wp]`` or any leading-axis
            variant of it.
        original_shape (Tuple[int, int]): ``(H, W)`` before padding.
        valid (np.ndarray): Boolean validity mask ``[H, W]``.
        spec (SarModelSpec): The model contract.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(label uint8 [H, W], probability float32
        [H, W])``. Labels use the model's class indices with ``spec.nodata`` at invalid
        pixels; probabilities are NaN there.
    """
    height, width = original_shape
    probability = np.asarray(water_prob, dtype=np.float32)
    while probability.ndim > 2:
        probability = probability[0]

    probability = np.array(probability[:height, :width], dtype=np.float32, copy=True)

    # Label with the model's own class indices instead of assuming water is 1. A
    # retrained model may declare {0: "water", 1: "land"}, and extraction decodes
    # grey_label through those same indices (merge_classes on
    # model_info.water_class_indices). Emitting a fixed 1 would invert the mask on such
    # a model rather than fail, which is exactly the desynchronization this module
    # exists to prevent.
    water_indices = spec.water_class_indices
    water_index = water_indices[0] if water_indices else 1
    land_index = next(
        (index for index in sorted(spec.classes) if index not in water_indices), 0
    )
    label = np.where(
        probability >= spec.water_threshold, water_index, land_index
    ).astype(np.uint8)
    label[~valid] = spec.nodata
    probability[~valid] = np.nan
    return label, probability


@dataclass(frozen=True)
class SarPrediction:
    """One scene's segmentation, georeferenced on the reference polarization's grid."""

    label: np.ndarray
    water_prob: np.ndarray
    valid: np.ndarray
    georef: np.ndarray
    projection: str


def segment_scene(
    file_paths: Sequence[str],
    spec: SarModelSpec,
    session: Optional["ort.InferenceSession"] = None,
) -> SarPrediction:
    """Segment one Sentinel-1 scene from its per-polarization GeoTIFFs.

    Args:
        file_paths (Sequence[str]): Per-polarization GeoTIFF paths for one scene.
        spec (SarModelSpec): The model contract.
        session (Optional[ort.InferenceSession]): Reusable session; created if omitted.

    Returns:
        SarPrediction: Labels, water probability and validity on the reference grid.
    """
    if session is None:
        session = get_session(spec)

    stack, valid, geotransform, projection = build_sar_stack(file_paths, spec)
    tensor, original_shape = preprocess(stack, valid, spec)
    outputs = session.run([spec.prob_output], {spec.input_name: tensor})
    label, probability = postprocess(outputs[0], original_shape, valid, spec)
    return SarPrediction(
        label=label,
        water_prob=probability,
        valid=valid,
        georef=np.asarray(geotransform, dtype=float),
        projection=projection,
    )


def sar_nodata_mask(file_paths: Sequence[str]) -> Optional[np.ndarray]:
    """Return the combined nodata mask for a scene, or None when it cannot be read.

    Sentinel-1 scenes are swath crops, so a large share of a downloaded tile is often
    empty. ``SDS_preprocess.preprocess_image`` reports an all-zero nodata mask for S1,
    which lets those blank areas reach the shoreline contour finder. This recovers the
    real mask from the GeoTIFFs.

    Args:
        file_paths (Sequence[str]): Per-polarization GeoTIFF paths for one scene.

    Returns:
        Optional[np.ndarray]: Boolean ``[H, W]`` mask, True where there is no data.
    """
    paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
    if not paths:
        return None
    try:
        _, valid, geotransform, projection = read_band(paths[0])
        for path in paths[1:]:
            _, other_valid = read_band_on_grid(
                path, geotransform, projection, valid.shape
            )
            valid = valid & other_valid
        return ~valid
    except Exception as error:
        logger.warning("Could not read the SAR nodata mask for %s: %s", paths, error)
        return None


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _label_to_colors(
    label: np.ndarray, nodata_mask: np.ndarray, class_count: int
) -> np.ndarray:
    """Colorize class indices with the shared palette; nodata pixels become black."""
    palette = [
        tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        for color in CLASS_LABEL_COLORMAP
    ]
    colored = np.zeros(label.shape[:2] + (3,), dtype=np.uint8)
    for class_index in range(max(class_count, 1)):
        colored[label == class_index] = palette[class_index % len(palette)]
    colored[nodata_mask] = (0, 0, 0)
    return colored


def save_prediction_png(
    prediction: SarPrediction, spec: SarModelSpec, output_path: Path
) -> Path:
    """Write the colorized ``*_predseg.png`` companion for a prediction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colored = _label_to_colors(
        prediction.label, ~prediction.valid, len(spec.classes) or 1
    )
    Image.fromarray(colored, mode="RGB").save(output_path)
    return output_path


def save_prediction_npz(
    prediction: SarPrediction,
    spec: SarModelSpec,
    input_file: str,
    output_path: Path,
) -> Path:
    """Write the ``*_res.npz`` artifact the shoreline extraction reads.

    ``grey_label`` carries the model's class indices with ``spec.nodata`` (255) at
    invalid pixels. That is safe downstream because ``merge_classes`` builds its binary
    mask with equality tests, so 255 simply is not water; nodata is excluded from
    contour finding through the nodata mask instead.

    ``nclasses`` is deliberately ``len(spec.classes)`` rather than ``label.max() + 1``,
    which would be 256 once the nodata code is present.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nclasses": len(spec.classes),
        "n_data_bands": len(spec.channel_order),
        "input_file": input_file,
        "grey_label": prediction.label.astype(np.uint8),
        "av_softmax_scores": prediction.water_prob.astype(np.float32),
        "classes_to_index": dict(spec.classes),
        # Extra provenance, ignored by the extraction path.
        "nodata_mask": ~prediction.valid,
        "water_threshold": np.float32(spec.water_threshold),
        "channel_order": list(spec.channel_order),
        "model_path": spec.onnx_path,
    }
    np.savez_compressed(str(output_path), **payload)
    return output_path


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_model_settings_json(
    session_directory: Path, model_directory: str, use_gpu: bool = False
) -> Path:
    """Write ``model_settings.json`` so the session can be resumed later."""
    return _write_json(
        session_directory / "model_settings.json",
        {
            "use_GPU": "1" if use_gpu else "0",
            "implementation": "BEST",
            "model_type": Path(model_directory).name,
            "img_type": "SAR",
        },
    )


def write_model_info_json(
    session_directory: Path,
    model_directory: str,
    input_directory: str,
    spec: SarModelSpec,
) -> Path:
    """Write ``model_info.json`` describing the class mapping and water classes."""
    water_class_indices = spec.water_class_indices
    return _write_json(
        session_directory / "model_info.json",
        {
            "model_directory": str(Path(model_directory).expanduser().resolve()),
            "input_directory": str(input_directory),
            "class_mapping": {str(k): v for k, v in spec.classes.items()},
            "water_class_indices": water_class_indices,
            "water_classes": [
                spec.classes[index]
                for index in water_class_indices
                if index in spec.classes
            ],
        },
    )


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def _scene_key(filename: str) -> str:
    """Group per-polarization files of the same acquisition under one key."""
    basename = os.path.basename(filename)
    polarization = get_polarization_from_filename(basename)
    if polarization:
        return _POLARIZATION_TOKEN_RE.sub("_POL", basename, count=1)
    return basename


def _output_stem(vv_filename: str) -> str:
    """Build the artifact stem, matching what the TensorFlow zoo path produces.

    ``{date}_RGB_S1`` mirrors the ``{date}_RGB_S1.jpg`` preview coastsat writes, which
    is what ``find_matching_npz`` and ``find_satellite_in_filename`` key off. Any
    ``_dupN`` suffix is preserved so duplicate acquisitions do not overwrite each other.
    """
    basename = os.path.basename(vv_filename)
    stem = f"{basename[:19]}_RGB_S1"
    duplicate = _DUP_TOKEN_RE.search(basename)
    if duplicate:
        stem = f"{stem}_{duplicate.group(1)}"
    return stem


def collect_sar_scenes(roi_directory: str) -> Dict[str, Dict[str, str]]:
    """Group the Sentinel-1 GeoTIFFs under an ROI directory by acquisition.

    Args:
        roi_directory (str): ROI data directory, e.g. ``data/ID_x_datetime...``.

    Returns:
        Dict[str, Dict[str, str]]: ``{scene_key: {polarization: path}}``.
    """
    s1_root = Path(roi_directory) / "S1"
    scenes: Dict[str, Dict[str, str]] = {}
    for polarization in get_sar_polarizations():
        folder = s1_root / polarization
        if not folder.is_dir():
            continue
        for tif in sorted(folder.glob("*.tif")):
            scenes.setdefault(_scene_key(tif.name), {})[polarization] = str(tif)
    return scenes


def run_sar_segmentation(
    *,
    roi_directory: str,
    session_directory: str,
    model_directory: str,
    filenames: Optional[Sequence[str]] = None,
    rgb_directory: str = "",
    overwrite: bool = True,
    use_gpu: bool = False,
    water_threshold: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Segment every dual-polarization Sentinel-1 scene in an ROI directory.

    Writes ``*_predseg.png`` and ``*_res.npz`` per scene plus ``model_settings.json``,
    ``model_info.json`` and ``segmentation_summary.json``, flat in ``session_directory``.

    Args:
        roi_directory (str): ROI data directory containing ``S1/<polarization>/``.
        session_directory (str): Where the artifacts are written.
        model_directory (str): Directory holding the exported ``.onnx`` model.
        filenames (Optional[Sequence[str]]): Restrict to these scene filenames
            (as listed in ``metadata['S1']['filenames']``). All scenes when omitted.
        rgb_directory (str): RGB preview directory recorded in ``model_info.json``.
        overwrite (bool): Re-run scenes whose artifacts already exist.
        use_gpu (bool): Try the CUDA execution provider before CPU.
        water_threshold (Optional[float]): ``P(water)`` cutoff. Overrides the value the
            model records in its own metadata; that value is used when this is None.
        progress_callback (Optional[Callable[[int, int, str], None]]): Called as
            ``(current, total, scene_name)`` after each scene.

    Returns:
        Dict[str, Any]: Summary with counts, failures and the session directory.

    Raises:
        MissingPolarizationError: If no scene has every polarization the model needs.
        SarModelError: If the model cannot be loaded.
    """
    session_path = Path(session_directory)
    session_path.mkdir(parents=True, exist_ok=True)

    spec = load_sar_model_spec(model_directory, use_gpu=use_gpu)
    if water_threshold is not None:
        # The user's setting wins over the cutoff baked into the model's metadata.
        spec = replace(spec, water_threshold=float(water_threshold))
    session = get_session(spec, use_gpu=use_gpu)
    required = spec.required_polarizations

    scenes = collect_sar_scenes(roi_directory)
    if filenames:
        wanted = {_scene_key(os.path.basename(name)) for name in filenames}
        scenes = {key: paths for key, paths in scenes.items() if key in wanted}

    complete = {
        key: paths
        for key, paths in scenes.items()
        if all(polarization in paths for polarization in required)
    }
    incomplete = {key: paths for key, paths in scenes.items() if key not in complete}

    if scenes and not complete:
        available = sorted({p for paths in scenes.values() for p in paths})
        raise MissingPolarizationError(
            f"The SAR model needs {list(required)} but "
            f"{os.path.join(roi_directory, 'S1')} only contains {available}. This site "
            f"was downloaded before multi-polarization support. "
            f"{describe_sar_polarization_remedy()}"
        )

    failures: List[Dict[str, str]] = []
    for key, paths in sorted(incomplete.items()):
        failures.append(
            {
                "image": key,
                "error": f"missing polarizations "
                f"{[p for p in required if p not in paths]}",
            }
        )
    if incomplete:
        logger.warning(
            "Skipping %d Sentinel-1 scene(s) that lack a required polarization.",
            len(incomplete),
        )

    written_masks = 0
    written_npz = 0
    skipped_existing = 0
    total = len(complete)

    for index, (key, paths) in enumerate(sorted(complete.items()), start=1):
        reference_path = paths[required[0]]
        stem = _output_stem(reference_path)
        png_path = session_path / f"{stem}_predseg.png"
        npz_path = session_path / f"{stem}_res.npz"

        try:
            if not overwrite and png_path.exists() and npz_path.exists():
                skipped_existing += 1
                continue

            ordered_paths = [paths[polarization] for polarization in required]
            prediction = segment_scene(ordered_paths, spec, session=session)

            valid_fraction = (
                float(prediction.valid.mean()) if prediction.valid.size else 0.0
            )
            logger.info("%s: %.1f%% of pixels are valid", stem, 100.0 * valid_fraction)

            save_prediction_png(prediction, spec, png_path)
            written_masks += 1
            save_prediction_npz(prediction, spec, reference_path, npz_path)
            written_npz += 1
        except Exception as error:
            logger.exception("SAR segmentation failed for %s", key)
            failures.append({"image": key, "error": str(error)})
        finally:
            if progress_callback is not None:
                progress_callback(index, total, stem)

    input_directory = rgb_directory or os.path.join(
        roi_directory, "jpg_files", "preprocessed", "RGB"
    )
    write_model_settings_json(session_path, model_directory, use_gpu=use_gpu)
    write_model_info_json(session_path, model_directory, input_directory, spec)

    summary: Dict[str, Any] = {
        "session_directory": str(session_path),
        "model_directory": str(Path(model_directory).expanduser().resolve()),
        "total_images": total,
        "written_masks": written_masks,
        "written_npz": written_npz,
        "skipped_existing": skipped_existing,
        "failures": failures,
    }
    _write_json(session_path / "segmentation_summary.json", summary)
    return summary
