"""Tests for the Sentinel-1 download settings and on-disk layout.

The SAR model needs both VV and VH, so these pin that the default requests both, that
the setting survives a config round-trip, and that the polarization folders are found.
"""

import json
import os

import pytest

from coastseg import common, extracted_shoreline


def make_selected_rois(*roi_ids):
    return {
        "features": [
            {
                "properties": {"id": roi_id},
                "geometry": {"coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]},
            }
            for roi_id in roi_ids
        ]
    }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_sentinel_1_properties_request_both_polarizations():
    properties = common.get_default_sentinel_1_properties()
    assert properties["transmitterReceiverPolarisation"] == ["VV", "VH"]
    assert properties["instrumentMode"] == "IW"


def test_create_roi_settings_defaults_to_vv_and_vh():
    roi_settings = common.create_roi_settings(
        settings={"sat_list": ["S1"], "dates": ["2023-05-01", "2023-06-01"]},
        selected_rois=make_selected_rois("1"),
        filepath="/data",
        date_str="01-01-25__00_00_00",
    )
    assert roi_settings["1"]["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VV", "VH"]


def test_create_roi_settings_gives_each_roi_its_own_properties_dict():
    """A shared mutable dict would let an edit to one ROI silently change every ROI."""
    roi_settings = common.create_roi_settings(
        settings={"sat_list": ["S1"], "dates": ["2023-05-01", "2023-06-01"]},
        selected_rois=make_selected_rois("1", "2"),
        filepath="/data",
        date_str="01-01-25__00_00_00",
    )

    first = roi_settings["1"]["sentinel_1_properties"]
    second = roi_settings["2"]["sentinel_1_properties"]
    assert first is not second

    first["transmitterReceiverPolarisation"].append("HH")
    assert second["transmitterReceiverPolarisation"] == ["VV", "VH"]


def test_create_roi_settings_honours_an_explicit_override():
    roi_settings = common.create_roi_settings(
        settings={
            "sat_list": ["S1"],
            "dates": ["2023-05-01", "2023-06-01"],
            "sentinel_1_properties": {
                "transmitterReceiverPolarisation": ["VH"],
                "instrumentMode": "IW",
            },
        },
        selected_rois=make_selected_rois("1"),
        filepath="/data",
        date_str="01-01-25__00_00_00",
    )
    assert roi_settings["1"]["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VH"]


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_extract_roi_settings_keeps_sentinel_1_properties():
    """Without this the setting is dropped whenever a session config is reloaded."""
    properties = {
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "instrumentMode": "IW",
    }
    json_data = {
        "roi_ids": ["1"],
        "1": {
            "dates": ["2023-05-01", "2023-06-01"],
            "sitename": "ID_1_datetime01-01-25__00_00_00",
            "polygon": [[[0, 0]]],
            "roi_id": "1",
            "sat_list": ["S1"],
            "landsat_collection": "C02",
            "filepath": "/data",
            "sentinel_1_properties": properties,
        },
    }

    roi_settings = common.extract_roi_settings(json_data)
    assert roi_settings["1"]["sentinel_1_properties"] == properties


def test_load_settings_keeps_sentinel_1_properties(tmp_path):
    properties = {
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "instrumentMode": "IW",
    }
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"sat_list": ["S1"], "sentinel_1_properties": properties}),
        encoding="utf-8",
    )

    settings = common.load_settings(str(config))
    assert settings["sentinel_1_properties"] == properties


def test_global_settings_upgrade_a_legacy_roi_without_the_key():
    """Legacy ROI settings have no sentinel_1_properties key at all.

    update_roi_settings skips keys that are absent, so without add_if_missing the ROI
    would stay on coastsat's VH-only fallback forever.
    """
    roi_settings = {"1": {"sat_list": ["S1"], "dates": ["2023-05-01", "2023-06-01"]}}
    global_settings = {
        "sat_list": ["S1"],
        "dates": ["2023-05-01", "2023-06-01"],
        "sentinel_1_properties": {
            "transmitterReceiverPolarisation": ["VV", "VH"],
            "instrumentMode": "IW",
        },
    }

    updated = common.update_roi_settings_with_global_settings(
        roi_settings, global_settings
    )
    assert updated["1"]["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VV", "VH"]


def test_global_settings_propagation_does_not_alias_the_global_dict():
    global_settings = {
        "sentinel_1_properties": {"transmitterReceiverPolarisation": ["VV", "VH"]}
    }
    roi_settings = {"1": {"sat_list": ["S1"]}}

    updated = common.update_roi_settings_with_global_settings(
        roi_settings, global_settings
    )
    updated["1"]["sentinel_1_properties"]["transmitterReceiverPolarisation"].append("HH")

    assert global_settings["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VV", "VH"]


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------


def test_get_filepath_returns_every_polarization_in_canonical_order(tmp_path):
    site = "ID_test_datetime01-01-25__00_00_00"
    # created out of order on purpose: the result must not depend on listdir order
    os.makedirs(tmp_path / site / "S1" / "VH")
    os.makedirs(tmp_path / site / "S1" / "VV")

    paths = extracted_shoreline.get_filepath(str(tmp_path), site, "S1")

    assert [os.path.basename(path) for path in paths] == ["VV", "VH"]


def test_get_filepath_on_a_legacy_vh_only_site(tmp_path):
    site = "ID_legacy_datetime01-01-20__00_00_00"
    os.makedirs(tmp_path / site / "S1" / "VH")

    paths = extracted_shoreline.get_filepath(str(tmp_path), site, "S1")

    assert [os.path.basename(path) for path in paths] == ["VH"]


def test_get_filepath_when_nothing_is_downloaded_yet(tmp_path):
    """Keep the legacy single-element shape so callers fail the way they always did."""
    paths = extracted_shoreline.get_filepath(str(tmp_path), "ID_missing", "S1")
    assert [os.path.basename(path) for path in paths] == ["VH"]
