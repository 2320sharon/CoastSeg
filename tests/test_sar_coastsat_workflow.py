"""Tests for the CoastSat (classic) Sentinel-1 workflow.

That workflow hands everything to ``SDS_shoreline.extract_shorelines``, which segments S1
with coastsat's trained ONNX model. coastsat reads three settings out of the dict CoastSeg
builds for it, so these tests pin two things:

1. the settings actually survive CoastSeg's allowlists and reach coastsat, and
2. ``sar_model_path`` is stored relative to the CoastSeg directory, never absolute, so a
   config.json stays usable on another machine.
"""

import os

import pytest

from coastseg import common, coastseg_map, sar_model
from coastseg import extracted_shoreline as es


SAR_KEYS = ("sar_segmentation", "sar_water_threshold", "sar_model_path")


@pytest.fixture
def based_at(monkeypatch, tmp_path):
    """Point core_utilities.get_base_dir at tmp_path for both path helpers."""
    monkeypatch.setattr(common.core_utilities, "get_base_dir", lambda *a, **k: tmp_path)
    return tmp_path


def shoreline_settings(settings):
    """Build the dict CoastSeg hands to SDS_shoreline.extract_shorelines."""
    return es.Extracted_Shoreline().create_shoreline_settings(
        settings,
        {"roi_id": "abc1", "sitename": "site", "filepath": "data"},
        reference_shoreline={},
    )


# ---------------------------------------------------------------------------
# The settings reach coastsat
# ---------------------------------------------------------------------------


def test_the_default_is_the_model_not_otsu():
    """The trained segmenter is the default path for Sentinel-1."""
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings()

    assert settings["sar_segmentation"] == "model"
    assert settings["sar_water_threshold"] == 0.5
    assert settings["sar_model_path"] == ""


@pytest.mark.parametrize(
    "key, value",
    [
        ("sar_segmentation", "otsu"),
        ("sar_water_threshold", 0.8),
    ],
)
def test_create_shoreline_settings_passes_the_sar_keys_to_coastsat(key, value):
    """coastsat reads these straight out of the dict; the allowlist used to drop them."""
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(**{key: value})

    assert shoreline_settings(settings)[key] == value


def test_empty_sar_model_path_survives_rather_than_being_stripped(monkeypatch):
    """'' is meaningful: it tells coastsat to resolve and download its own model.

    It must reach coastsat rather than being dropped by the SHORELINE_KEYS allowlist.
    With no model in CoastSeg's models/ directory it stays '' -- see
    test_an_unset_path_points_coastsat_at_a_manually_downloaded_model for the case
    where one is there.
    """
    monkeypatch.setattr(sar_model, "find_local_sar_model", lambda: "")
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings()

    assert shoreline_settings(settings)["sar_model_path"] == ""


def test_an_unset_path_points_coastsat_at_a_manually_downloaded_model(monkeypatch):
    """A user who downloads the model by hand into models/SAR_segmentation_model
    expects both workflows to use it. The zoo path reads that directory directly, but
    coastsat looks only in its own pooch cache -- so without this the same 130 MB would
    be downloaded a second time for the coastsat run."""
    manual = r"C:\CoastSeg\models\SAR_segmentation_model\SAR_3_band_model.onnx"
    monkeypatch.setattr(sar_model, "find_local_sar_model", lambda: manual)
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings()

    assert shoreline_settings(settings)["sar_model_path"] == manual


def test_an_explicit_path_is_not_overridden_by_the_local_model(monkeypatch):
    """An explicit choice always wins; the models/ directory is only a fallback."""
    monkeypatch.setattr(
        sar_model, "find_local_sar_model", lambda: r"C:\CoastSeg\models\local.onnx"
    )
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_model_path="models/retrained.onnx"
    )

    handed_to_coastsat = shoreline_settings(settings)["sar_model_path"]
    assert handed_to_coastsat.endswith(os.path.join("models", "retrained.onnx"))
    assert "local.onnx" not in handed_to_coastsat


def test_settings_absent_from_the_input_are_simply_absent():
    """A settings dict predating these keys must not gain them with junk values."""
    result = shoreline_settings({"output_epsg": 4326, "min_beach_area": 100})

    assert not any(key in result for key in SAR_KEYS)


def test_the_override_reaches_extract_shorelines(monkeypatch, tmp_path):
    """End to end through Extracted_Shoreline.extract_shorelines.

    A unit test on SHORELINE_KEYS alone would not catch a later refactor that rebuilds
    the settings dict somewhere else, so capture what coastsat is actually handed.
    """
    captured = {}

    def fake_extract_shorelines(metadata, settings, **kwargs):
        captured.update(settings)
        return {}

    monkeypatch.setattr(
        es.SDS_shoreline, "extract_shorelines", fake_extract_shorelines
    )
    monkeypatch.setattr(es, "get_metadata", lambda *a, **k: {"S1": {"filenames": ["x"]}})
    monkeypatch.setattr(es.common, "filter_metadata_with_dates", lambda m, *a, **k: m)
    monkeypatch.setattr(es, "get_reference_shoreline", lambda *a, **k: {})
    monkeypatch.setattr(es.SDS_tools, "remove_duplicates", lambda output: output)
    monkeypatch.setattr(es.SDS_tools, "remove_inaccurate_georef", lambda output, _: output)
    monkeypatch.setattr(es, "get_data_folder", lambda *a, **k: str(tmp_path))

    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_segmentation="otsu", sar_water_threshold=0.75
    )
    es.Extracted_Shoreline().extract_shorelines(
        shoreline_gdf=None,
        roi_settings={
            "roi_id": "abc1",
            "sitename": "site",
            "filepath": str(tmp_path),
            "sat_list": ["S1"],
            "dates": ["2023-01-01", "2024-01-01"],
        },
        settings=settings,
    )

    assert captured["sar_segmentation"] == "otsu"
    assert captured["sar_water_threshold"] == 0.75


def test_the_sar_keys_round_trip_through_load_settings(tmp_path):
    written = {
        "sar_segmentation": "otsu",
        "sar_water_threshold": 0.7,
        "sar_model_path": "models/retrained.onnx",
    }
    loaded = common.load_settings(new_settings=written)

    assert {key: loaded[key] for key in SAR_KEYS} == written


def test_the_sar_keys_round_trip_through_extract_roi_settings():
    config = {
        "roi_ids": ["abc1"],
        "abc1": {
            "sitename": "site",
            "sat_list": ["S1"],
            "sar_segmentation": "otsu",
            "sar_water_threshold": 0.7,
            "sar_model_path": "models/retrained.onnx",
        },
    }
    roi_settings = common.extract_roi_settings(config)

    assert roi_settings["abc1"]["sar_segmentation"] == "otsu"
    assert roi_settings["abc1"]["sar_water_threshold"] == 0.7
    assert roi_settings["abc1"]["sar_model_path"] == "models/retrained.onnx"


# ---------------------------------------------------------------------------
# sar_model_path is stored relative, resolved absolute
# ---------------------------------------------------------------------------


def test_empty_path_stays_empty_through_both_helpers(based_at):
    """coastsat falls back to its packaged model on a falsy value."""
    assert common.relativize_sar_model_path("") == ""
    assert common.resolve_sar_model_path("") == ""


def test_an_absolute_path_inside_the_tree_is_stored_relative(based_at):
    absolute = str(based_at / "models" / "retrained.onnx")

    stored = common.relativize_sar_model_path(absolute)

    assert stored == "models/retrained.onnx"  # POSIX, so a Windows config loads on Linux
    assert not os.path.isabs(stored)
    assert os.path.normcase(common.resolve_sar_model_path(stored)) == os.path.normcase(
        absolute
    )


def test_relativizing_is_idempotent(based_at):
    stored = common.relativize_sar_model_path(str(based_at / "models" / "m.onnx"))

    assert common.relativize_sar_model_path(stored) == stored


def test_a_relative_path_resolves_against_the_base_dir_not_the_cwd(
    based_at, monkeypatch, tmp_path
):
    """coastsat runs os.path.abspath on whatever it is given, which would resolve
    against wherever the notebook happens to be running."""
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = common.resolve_sar_model_path("models/retrained.onnx")

    assert os.path.normcase(resolved) == os.path.normcase(
        str(based_at / "models" / "retrained.onnx")
    )
    assert os.path.normcase(str(elsewhere)) not in os.path.normcase(resolved)


def test_a_path_outside_the_tree_is_kept_verbatim_and_warns(based_at, tmp_path, caplog):
    outside = tmp_path.parent / "outside_the_repo" / "m.onnx"

    with caplog.at_level("WARNING"):
        stored = common.relativize_sar_model_path(str(outside))

    assert stored == outside.as_posix()
    assert "cannot be stored relatively" in caplog.text


def test_set_settings_never_persists_an_absolute_path(based_at):
    absolute = str(based_at / "models" / "retrained.onnx")

    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_model_path=absolute
    )

    assert settings["sar_model_path"] == "models/retrained.onnx"


def test_coastsat_receives_the_absolute_path(based_at):
    """Stored relative for portability, handed to coastsat absolute so it can be opened."""
    absolute = based_at / "models" / "retrained.onnx"
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_model_path=str(absolute)
    )

    handed_to_coastsat = shoreline_settings(settings)["sar_model_path"]

    assert os.path.isabs(handed_to_coastsat)
    assert os.path.normcase(handed_to_coastsat) == os.path.normcase(str(absolute))
