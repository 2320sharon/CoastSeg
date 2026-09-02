"""Tests for the artifacts the SAR path writes into a session directory.

Shoreline extraction is model-agnostic: it only knows how to read `_res.npz` files and
`model_info.json`. These tests pin that contract so the SAR segmenter's output stays
consumable without any changes downstream.

Inference is stubbed, so none of this loads the real 130 MB model.
"""

import json
import os

import numpy as np
import pytest

from coastseg import common, extracted_shoreline, sar_model
from coastseg.model_info import ModelInfo


# Column boundary the stub predicts water to the left of. Chosen to sit inside the
# unpadded 70-pixel-wide fixture so the assertions do not depend on the padded width.
WATER_COLUMNS = 20


class StubSession:
    """Stands in for an onnxruntime session: water left of WATER_COLUMNS, land right."""

    def __init__(self):
        self.calls = []

    def run(self, output_names, feed):
        (tensor,) = feed.values()
        self.calls.append(tensor.shape)
        _, _, height, width = tensor.shape
        probability = np.zeros((1, 1, height, width), dtype=np.float32)
        probability[:, :, :, :WATER_COLUMNS] = 0.99
        return [probability]


@pytest.fixture
def stubbed_sar(monkeypatch, sar_model_spec_stub):
    """Run the batch driver without coastsat or the real model."""
    session = StubSession()
    monkeypatch.setattr(
        sar_model, "load_sar_model_spec", lambda *a, **k: sar_model_spec_stub
    )
    monkeypatch.setattr(sar_model, "get_session", lambda *a, **k: session)
    return session


def run_segmentation(tmp_path, site, **kwargs):
    return sar_model.run_sar_segmentation(
        roi_directory=site,
        session_directory=str(tmp_path / "session"),
        model_directory=str(tmp_path / "model"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_writes_the_expected_artifact_set(tmp_path, sar_scene_tifs, stubbed_sar):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    written = sorted(os.listdir(summary["session_directory"]))

    assert written == [
        "2023-05-01-04-26-29_RGB_S1_predseg.png",
        "2023-05-01-04-26-29_RGB_S1_res.npz",
        "model_info.json",
        "model_settings.json",
        "segmentation_summary.json",
    ]
    assert summary["total_images"] == 1
    assert summary["written_npz"] == 1
    assert summary["failures"] == []


def test_npz_keys_and_dtypes(tmp_path, sar_scene_tifs, stubbed_sar):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    npz_path = os.path.join(
        summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"
    )
    data = np.load(npz_path, allow_pickle=True)

    assert data["grey_label"].dtype == np.uint8
    assert set(np.unique(data["grey_label"]).tolist()) <= {0, 1, 255}
    assert data["grey_label"].shape == sar_scene_tifs["shape"]
    assert int(data["n_data_bands"]) == 3
    assert data["classes_to_index"].item() == {0: "land", 1: "water"}


def test_nclasses_is_the_class_count_not_the_max_label(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    """Deriving nclasses from label.max() + 1 would give 256 once nodata is present."""
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    data = np.load(
        os.path.join(summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"),
        allow_pickle=True,
    )

    assert int(data["nclasses"]) == 2
    assert 255 in np.unique(data["grey_label"])


def test_probability_is_stored_with_nan_at_nodata(tmp_path, sar_scene_tifs, stubbed_sar):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    data = np.load(
        os.path.join(summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"),
        allow_pickle=True,
    )
    nodata_mask = data["nodata_mask"]

    assert data["av_softmax_scores"].dtype == np.float32
    assert np.all(np.isnan(data["av_softmax_scores"][nodata_mask]))
    assert not np.any(np.isnan(data["av_softmax_scores"][~nodata_mask]))


def test_predseg_png_is_written_for_the_segmentation_filter(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    """apply_segmentation_filter globs for images and pairs them with the npz."""
    from PIL import Image

    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    png_path = os.path.join(
        summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_predseg.png"
    )
    image = np.array(Image.open(png_path))

    assert image.shape == sar_scene_tifs["shape"] + (3,)
    # nodata rows are rendered black
    assert np.all(image[-sar_scene_tifs["nodata_rows"] :, :, :] == 0)


# ---------------------------------------------------------------------------
# The water threshold setting
# ---------------------------------------------------------------------------


class GradedSession:
    """Predicts a middling P(water) so the cutoff decides the label, not the model."""

    def run(self, output_names, feed):
        (tensor,) = feed.values()
        _, _, height, width = tensor.shape
        probability = np.full((1, 1, height, width), 0.6, dtype=np.float32)
        return [probability]


@pytest.fixture
def graded_sar(monkeypatch, sar_model_spec_stub):
    monkeypatch.setattr(
        sar_model, "load_sar_model_spec", lambda *a, **k: sar_model_spec_stub
    )
    monkeypatch.setattr(sar_model, "get_session", lambda *a, **k: GradedSession())


def water_fraction_at(tmp_path, site, session_name, **kwargs):
    """Segment the site into its own session and return (water fraction, threshold)."""
    summary = sar_model.run_sar_segmentation(
        roi_directory=site,
        session_directory=str(tmp_path / session_name),
        model_directory=str(tmp_path / "model"),
        **kwargs,
    )
    data = np.load(
        os.path.join(summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"),
        allow_pickle=True,
    )
    label = data["grey_label"]
    valid = label != 255
    return float(np.mean(label[valid] == 1)), float(data["water_threshold"])


def test_the_model_metadata_threshold_is_used_when_none_is_supplied(
    tmp_path, sar_scene_tifs, graded_sar
):
    fraction, threshold = water_fraction_at(tmp_path, sar_scene_tifs["site"], "default")

    assert threshold == pytest.approx(0.5)  # stored float32, so compare loosely
    assert fraction == 1.0  # P(water)=0.6 clears a 0.5 cutoff everywhere


def test_the_supplied_threshold_overrides_the_model_metadata(
    tmp_path, sar_scene_tifs, graded_sar
):
    """The user's sar_water_threshold has to reach the batch path.

    It used to be dropped on the way through Zoo_Model.run_model, leaving the cutoff
    pinned at whatever the .onnx recorded while the npz claimed otherwise.
    """
    fraction, threshold = water_fraction_at(
        tmp_path, sar_scene_tifs["site"], "strict", water_threshold=0.8
    )

    assert threshold == pytest.approx(0.8)
    assert fraction == 0.0  # the same P(water)=0.6 now falls short


def test_zoo_model_forwards_the_water_threshold_setting(monkeypatch, tmp_path):
    """Zoo_Model.run_model must pass the setting down rather than dropping it."""
    captured = {}
    monkeypatch.setattr(
        sar_model,
        "run_sar_segmentation",
        lambda **kwargs: captured.update(kwargs)
        or {"written_npz": 0, "total_images": 0, "failures": []},
    )

    from coastseg import zoo_model as zoo

    model = zoo.Zoo_Model()
    monkeypatch.setattr(
        model, "resolve_model_directory", lambda setting: str(tmp_path / "sar_dir")
    )
    monkeypatch.setattr(sar_model, "is_sar_model_directory", lambda directory: True)

    model.run_model(
        {"sar_water_threshold": 0.75},
        "sample_directory",
        "session_directory",
        roi_directory="roi",
    )

    assert captured["water_threshold"] == 0.75


# ---------------------------------------------------------------------------
# Downstream consumers
# ---------------------------------------------------------------------------


def test_load_merged_image_labels_returns_the_water_mask(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    npz_path = os.path.join(
        summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"
    )

    water = extracted_shoreline.load_merged_image_labels(npz_path, class_indices=[1])
    height, width = sar_scene_tifs["shape"]
    valid_rows = height - sar_scene_tifs["nodata_rows"]

    assert water.shape == (height, width)
    assert water[:valid_rows, :WATER_COLUMNS].all()
    assert not water[:valid_rows, WATER_COLUMNS:].any()
    # nodata must never be counted as water
    assert not water[valid_rows:, :].any()


def test_find_matching_npz_resolves_from_the_vv_filename(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    match = extracted_shoreline.find_matching_npz(
        os.path.basename(sar_scene_tifs["vv"]), summary["session_directory"]
    )
    assert match is not None
    assert match.endswith("2023-05-01-04-26-29_RGB_S1_res.npz")


def test_get_filtered_dates_dict_sees_the_sar_session(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    dates = common.get_filtered_dates_dict(summary["session_directory"], "npz")
    assert dates["S1"] == {"2023-05-01-04-26-29"}


def test_model_info_json_round_trips_through_model_info(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(
        tmp_path, sar_scene_tifs["site"], rgb_directory=sar_scene_tifs["rgb_directory"]
    )
    info = ModelInfo(
        model_info_path=os.path.join(summary["session_directory"], "model_info.json")
    )
    info.load_from_model_info_file()

    assert info.class_mapping == {0: "land", 1: "water"}
    assert info.water_class_indices == [1]
    assert info.input_directory == sar_scene_tifs["rgb_directory"]


def test_model_settings_json_records_the_sar_image_type(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    settings = json.loads(
        open(
            os.path.join(summary["session_directory"], "model_settings.json"),
            encoding="utf-8",
        ).read()
    )
    assert settings["img_type"] == "SAR"
    assert settings["model_type"] == "model"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_vh_only_site_names_the_missing_polarizations(
    tmp_path, sar_site_vh_only, stubbed_sar
):
    with pytest.raises(sar_model.MissingPolarizationError) as error:
        run_segmentation(tmp_path, sar_site_vh_only["site"])

    assert "['VV', 'VH']" in str(error.value)


def test_legacy_vh_only_advice_says_to_re_run_the_download(
    tmp_path, sar_site_vh_only, stubbed_sar
):
    """coastsat backfills the missing band on the next download, so the error tells the
    user to re-run it. Telling them to re-download to a new sitename instead would cost
    them a full download for nothing."""
    with pytest.raises(sar_model.MissingPolarizationError) as error:
        run_segmentation(tmp_path, sar_site_vh_only["site"])

    message = str(error.value).lower()
    assert "re-run the download" in message
    assert "new sitename" not in message


def test_scenes_missing_a_polarization_are_recorded_not_fatal(
    tmp_path, sar_scene_tifs, sar_tif_writer, stubbed_sar
):
    """A site with one complete and one incomplete scene segments what it can."""
    build_sar_bands, write_sar_tif = sar_tif_writer

    _, vh = build_sar_bands(nodata_rows=0)
    write_sar_tif(
        os.path.join(
            sar_scene_tifs["site"], "S1", "VH", "2024-06-02-04-26-29_S1_test_VH.tif"
        ),
        vh,
    )

    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])

    assert summary["total_images"] == 1
    assert summary["written_npz"] == 1
    assert len(summary["failures"]) == 1
    assert "VV" in summary["failures"][0]["error"]


def test_overwrite_false_skips_existing_artifacts(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    run_segmentation(tmp_path, sar_scene_tifs["site"])
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"], overwrite=False)

    assert summary["skipped_existing"] == 1
    assert summary["written_npz"] == 0


def test_filenames_argument_restricts_the_scenes_processed(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(
        tmp_path, sar_scene_tifs["site"], filenames=["1999-01-01-00-00-00_S1_test_VV.tif"]
    )
    assert summary["total_images"] == 0
