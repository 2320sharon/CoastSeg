"""Tests that Sentinel-1 shorelines can be extracted in both workflows.

The two workflows reach segmentation by different routes:

* **zoo** -- ``Zoo_Model.run_model`` segments first and writes ``*_res.npz``, then
  ``extract_shorelines_with_dask`` traces contours from the class labels. The SAR branch
  runs onnxruntime in-process; the optical branch shells out to the TensorFlow
  environment. Picking the wrong one is silent until it fails much later.
* **coastsat** -- ``Extracted_Shoreline.extract_shorelines`` hands everything to
  ``SDS_shoreline.extract_shorelines``, which segments S1 itself.

Nothing here loads the 130 MB ONNX model, launches the TensorFlow subprocess, or reads a
real scene: the two segmenters and coastsat's extractor are replaced by capturing stubs.
"""

import numpy as np
import pytest

from coastseg import extracted_shoreline as es
from coastseg import sar_model, zoo_model


# ---------------------------------------------------------------------------
# Zoo workflow: which segmenter runs
# ---------------------------------------------------------------------------


@pytest.fixture
def routed_segmenters(monkeypatch):
    """Stub both segmentation backends and record which one was called."""
    calls = {}

    def fake_sar_segmentation(**kwargs):
        calls["sar"] = kwargs
        return {"written_npz": 1, "total_images": 1, "failures": []}

    def fake_external_segmentation(**kwargs):
        calls["external"] = kwargs

    monkeypatch.setattr(sar_model, "run_sar_segmentation", fake_sar_segmentation)
    monkeypatch.setattr(
        zoo_model, "run_segmentation_in_external_env", fake_external_segmentation
    )
    return calls


def model_directory(tmp_path, name, weights):
    """A model directory identified only by the extension of the file inside it."""
    directory = tmp_path / name
    directory.mkdir()
    (directory / weights).write_bytes(b"")
    return str(directory)


def run_model(monkeypatch, directory, roi_directory="roi"):
    """Invoke Zoo_Model.run_model against a directory, skipping model resolution."""
    model = zoo_model.Zoo_Model()
    monkeypatch.setattr(model, "resolve_model_directory", lambda setting: directory)
    model.run_model(
        {}, "sample_directory", "session_directory", roi_directory=roi_directory
    )


def test_a_sar_model_directory_routes_to_the_onnx_driver(
    monkeypatch, tmp_path, routed_segmenters
):
    """A directory holding an .onnx is a SAR model and runs in-process."""
    directory = model_directory(tmp_path, "sar_model_dir", "SAR_3_band_model.onnx")

    run_model(monkeypatch, directory, roi_directory="roi_data")

    assert "external" not in routed_segmenters
    assert routed_segmenters["sar"]["roi_directory"] == "roi_data"
    assert routed_segmenters["sar"]["model_directory"] == directory


def test_an_optical_model_directory_routes_to_the_tensorflow_env(
    monkeypatch, tmp_path, routed_segmenters
):
    """The complement: no .onnx means the SegFormer path in the separate environment.

    Together with the test above this pins is_sar_model_directory as the whole
    discriminator -- there is no img_type flag consulted here.
    """
    directory = model_directory(tmp_path, "segformer_dir", "weights.h5")

    run_model(monkeypatch, directory)

    assert "sar" not in routed_segmenters
    assert routed_segmenters["external"]["model_path"] == directory


def test_the_sar_path_requires_the_roi_directory(
    monkeypatch, tmp_path, routed_segmenters
):
    """The SAR model reads the S1 GeoTIFFs, not the jpg previews the optical path uses,
    so a missing ROI directory has to fail loudly rather than segment nothing."""
    directory = model_directory(tmp_path, "sar_model_dir", "model.onnx")

    with pytest.raises(ValueError, match="ROI directory"):
        run_model(monkeypatch, directory, roi_directory="")

    assert routed_segmenters == {}


# ---------------------------------------------------------------------------
# Zoo workflow: reaching the SAR model without naming its directory
# ---------------------------------------------------------------------------


@pytest.fixture
def designated_sar_directory(monkeypatch, tmp_path):
    """Point CoastSeg's designated SAR directory somewhere empty and unwritten.

    Absent, not merely empty: the model is 130 MB and downloaded on first use, so a
    fresh clone has no such directory at all.
    """
    directory = tmp_path / "models" / sar_model.DEFAULT_SAR_MODEL_DIRNAME
    monkeypatch.setattr(
        zoo_model.sar_model, "get_sar_model_directory", lambda: str(directory)
    )
    monkeypatch.setattr(
        zoo_model.sar_model, "load_sar_model_spec", lambda directory: None
    )
    return directory


def test_selecting_sar_without_a_local_path_uses_the_designated_directory(
    monkeypatch, designated_sar_directory
):
    """``img_type='SAR'`` alone is enough to reach the model.

    'SAR' names an image type, not a Zoo model id, so falling through to the Zoo
    downloader looks ``model_type`` up on Zenodo and fails with a 404 -- an error that
    says nothing about SAR. The UI never sees this because it fills in
    'local_model_path' itself; a script that sets only 'img_type' does.
    """

    def fail(*args, **kwargs):  # pragma: no cover - only runs if the branch is wrong
        raise AssertionError("SAR must not be resolved through the Zoo downloader")

    model = zoo_model.Zoo_Model()
    monkeypatch.setattr(model, "ensure_model_downloaded", fail)

    resolved = model.resolve_model_directory(
        {"img_type": "SAR", "model_type": "SAR_segmentation_model"}
    )

    assert resolved == str(designated_sar_directory)


def test_the_sar_directory_resolves_before_the_model_is_downloaded(
    designated_sar_directory,
):
    """An absent SAR directory is the fresh-clone state, not a configuration error.

    ``is_sar_model_directory`` deliberately answers True here so the first run routes
    to the ONNX path and downloads the model. Rejecting the path for not existing yet
    would make that download unreachable.
    """
    assert not designated_sar_directory.exists()

    model = zoo_model.Zoo_Model()
    resolved = model.resolve_model_directory(
        {"img_type": "SAR", "local_model_path": str(designated_sar_directory)}
    )

    assert resolved == str(designated_sar_directory)


def test_a_missing_optical_model_path_is_still_an_error(tmp_path):
    """The tolerance above is scoped to SAR; a typo'd optical path must still raise."""
    missing = tmp_path / "not_a_real_model_directory"

    model = zoo_model.Zoo_Model()
    with pytest.raises(FileNotFoundError, match="local_model_path"):
        model.resolve_model_directory(
            {"img_type": "RGB", "local_model_path": str(missing)}
        )


def test_a_missing_non_sar_directory_is_an_error_even_when_sar_is_selected(tmp_path):
    """Selecting SAR does not excuse a path that is not a SAR directory at all."""
    missing = tmp_path / "some_other_model"

    model = zoo_model.Zoo_Model()
    with pytest.raises(FileNotFoundError, match="local_model_path"):
        model.resolve_model_directory(
            {"img_type": "SAR", "local_model_path": str(missing)}
        )


# ---------------------------------------------------------------------------
# Zoo workflow: where the class mapping comes from
# ---------------------------------------------------------------------------


@pytest.fixture
def sar_spec(monkeypatch):
    """Stub the ONNX spec so no 130 MB model is loaded."""

    class Spec:
        classes = {0: "land", 1: "water"}

    monkeypatch.setattr(sar_model, "load_sar_model_spec", lambda directory: Spec())
    return Spec


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(["SAR_3_band_model.onnx"], id="downloaded-onnx-only"),
        pytest.param(["my_retrained.onnx"], id="user-supplied-onnx"),
    ],
)
def test_a_sar_model_directory_takes_its_classes_from_the_onnx(
    monkeypatch, tmp_path, sar_spec, contents
):
    """A SAR model directory has no ``*modelcard.json``, and must not be given the
    optical default mapping because of it.

    The model is fetched as a bare ``.onnx`` into coastsat's cache, and a user supplying
    their own drops in one file, so neither case ever produces a model card. Falling back
    to the 4-class Zoo mapping sets ``water_class_indices`` to ``[0, 1]``, which on SAR's
    ``{0: land, 1: water}`` labels marks *every* pixel as water: no land/water boundary,
    no contour, and extraction raises ``No_Extracted_Shoreline`` while segmentation looks
    perfectly healthy.
    """
    directory = tmp_path / "SAR_segmentation_model"
    directory.mkdir()
    for name in contents:
        (directory / name).write_bytes(b"")
    assert not list(directory.glob("*modelcard.json"))

    model = zoo_model.Zoo_Model()
    info = model.load_model_info({"local_model_path": str(directory)})

    assert info.class_mapping == {0: "land", 1: "water"}
    assert info.water_class_indices == [1]


def test_an_optical_model_directory_still_reads_its_model_card(monkeypatch, tmp_path):
    """The SAR branch must not intercept a directory holding ordinary Zoo weights."""
    directory = tmp_path / "segformer_RGB_4class"
    directory.mkdir()
    (directory / "weights.h5").write_bytes(b"")

    def fail(directory):  # pragma: no cover - only runs if the branch is wrong
        raise AssertionError("the optical path must not load a SAR spec")

    monkeypatch.setattr(sar_model, "load_sar_model_spec", fail)

    model = zoo_model.Zoo_Model()
    info = model.load_model_info({"local_model_path": str(directory)})

    # No model card either, so this is the documented optical default.
    assert info.water_class_indices == [0, 1]


# ---------------------------------------------------------------------------
# Zoo workflow: extraction from a segmented session
# ---------------------------------------------------------------------------


def test_the_zoo_session_tags_s1_shorelines_as_sar_model(monkeypatch, tmp_path):
    """S1 goes through extraction and is recorded as segmented by the SAR model.

    ``segmentation_method`` is what later tells a threshold filter to leave these
    shorelines alone, and ``combine_satellite_data`` builds its column list from the keys
    present, so a column missing on one satellite desynchronizes the merged lists.
    """
    processed = []

    def fake_process_satellite(satname, settings, metadata, session_path, *args, **kwargs):
        processed.append(satname)
        return {
            satname: {
                "dates": ["2023-05-01-04-26-29"],
                "shorelines": [np.array([[0.0, 0.0], [1.0, 1.0]])],
                "filename": ["2023-05-01-04-26-29_S1_site_VV.tif"],
                "cloud_cover": [0.0],
                "geoaccuracy": [1.0],
                "idx": [0],
                "satname": [satname],
                "segmentation_method": [es.get_segmentation_method(satname)],
            }
        }

    monkeypatch.setattr(es, "process_satellite", fake_process_satellite)

    output = es.extract_shorelines_with_dask(
        session_path=str(tmp_path),
        metadata={"S1": {"filenames": ["a"], "dates": ["2023-05-01-04-26-29"]}},
        settings={"inputs": {"sitename": "site", "filepath": str(tmp_path)}},
        save_location=str(tmp_path),
    )

    assert processed == ["S1"]
    assert output["satname"] == ["S1"]
    assert output["segmentation_method"] == ["sar_model"]
    assert len(output["segmentation_method"]) == len(output["shorelines"])


# ---------------------------------------------------------------------------
# CoastSat workflow
# ---------------------------------------------------------------------------


def test_the_coastsat_workflow_hands_coastsat_an_s1_only_run(monkeypatch, tmp_path):
    """coastsat decides to segment with its SAR model from the run it is handed.

    ``sat_list`` and the metadata keys are what select the S1 branch inside
    ``SDS_shoreline.extract_shorelines``; an ROI that loses either of them silently falls
    through to the optical path.
    """
    captured = {}

    def fake_extract_shorelines(metadata, settings, **kwargs):
        captured["metadata"] = metadata
        captured["settings"] = settings
        return {}

    monkeypatch.setattr(es.SDS_shoreline, "extract_shorelines", fake_extract_shorelines)
    monkeypatch.setattr(
        es, "get_metadata", lambda *a, **k: {"S1": {"filenames": ["scene_VV.tif"]}}
    )
    monkeypatch.setattr(es.common, "filter_metadata_with_dates", lambda m, *a, **k: m)
    monkeypatch.setattr(es, "get_reference_shoreline", lambda *a, **k: {})
    monkeypatch.setattr(es.SDS_tools, "remove_duplicates", lambda output: output)
    monkeypatch.setattr(
        es.SDS_tools, "remove_inaccurate_georef", lambda output, _: output
    )
    monkeypatch.setattr(es, "get_data_folder", lambda *a, **k: str(tmp_path))

    es.Extracted_Shoreline().extract_shorelines(
        shoreline_gdf=None,
        roi_settings={
            "roi_id": "abc1",
            "sitename": "site",
            "filepath": str(tmp_path),
            "sat_list": ["S1"],
            "dates": ["2023-01-01", "2024-01-01"],
        },
        settings={"output_epsg": 4326, "min_beach_area": 100},
    )

    assert list(captured["metadata"]) == ["S1"]
    assert captured["settings"]["inputs"]["sat_list"] == ["S1"]


def test_a_sar_shoreline_is_exempt_from_the_mndwi_threshold_filter():
    """A SAR shoreline carries no MNDWI threshold, so it must not be filtered on one.

    ``MNDWI_threshold`` holds a different quantity per segmentation method: an MNDWI value
    for the optical contour path, but NaN when a model mapped the shoreline. Comparing NaN
    to an MNDWI range fails every time and would delete every SAR shoreline.
    """
    output = {
        "shorelines": [np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]])],
        "MNDWI_threshold": [float("nan"), 0.1],
        "segmentation_method": ["sar_model", "mndwi"],
    }

    applies = es.threshold_filter_applies(output)

    assert list(applies) == [False, True]
