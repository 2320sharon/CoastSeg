"""Tests for the Sentinel-1 download and the resume of a partly-downloaded site.

The download itself lives in coastsat; CoastSeg only builds the ``inputs`` block and calls
``SDS_download.retrieve_images``. These tests pin that boundary:

1. an S1 download asks for both polarizations, and an explicit choice is not overwritten,
2. a site missing a polarization reaches coastsat untouched -- CoastSeg does not intercept
   or warn, because coastsat backfills it during this same call and reports that itself,
3. re-running such a site re-queues the incomplete scene and backfills VV in place --
   same sitename, no duplicate.

Earth Engine is never contacted: ``retrieve_images`` is replaced by a capturing stub.
"""

import os

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from coastsat import SDS_download

from coastseg import coastseg_map
from coastseg import extracted_shoreline as es


@pytest.fixture
def captured_downloads(monkeypatch):
    """Replace coastsat's downloader with a stub that records what it was handed."""
    calls = []
    monkeypatch.setattr(
        SDS_download,
        "retrieve_images",
        lambda inputs, **kwargs: calls.append(inputs),
    )
    return calls


@pytest.fixture
def one_roi_gdf():
    """A single ROI, the shape CoastSeg_Map.download_imagery expects."""
    return gpd.GeoDataFrame(
        {"id": ["roi1"]},
        geometry=[
            Polygon(
                [
                    (-122.50, 37.70),
                    (-122.40, 37.70),
                    (-122.40, 37.80),
                    (-122.50, 37.80),
                ]
            )
        ],
        crs="EPSG:4326",
    )


def download_one_roi(monkeypatch, roi_gdf, destination, **settings):
    """Run CoastSeg_Map.download_imagery for one S1 ROI without writing a config."""
    coastseg_object = coastseg_map.CoastSeg_Map(create_map=False)
    monkeypatch.setattr(coastseg_object, "save_config", lambda *a, **k: None)
    resolved = coastseg_object.set_settings(
        sat_list=["S1"],
        dates=["2023-01-01", "2023-06-01"],
        landsat_collection="C02",
        **settings,
    )
    coastseg_object.download_imagery(
        rois=roi_gdf,
        settings=resolved,
        selected_ids={"roi1"},
        file_path=str(destination),
    )


# ---------------------------------------------------------------------------
# Both polarizations are requested
# ---------------------------------------------------------------------------


def test_download_imagery_asks_coastsat_for_both_polarizations(
    monkeypatch, tmp_path, one_roi_gdf, captured_downloads
):
    """The SAR segmentation model needs VV, VH and their difference.

    A download that quietly asks for one polarization produces a site the model cannot
    read, and the failure only shows up much later at extraction time.
    """
    download_one_roi(monkeypatch, one_roi_gdf, tmp_path)

    assert len(captured_downloads) == 1
    inputs = captured_downloads[0]
    assert inputs["sat_list"] == ["S1"]
    assert inputs["sentinel_1_properties"]["transmitterReceiverPolarisation"] == [
        "VV",
        "VH",
    ]


def test_an_explicit_polarization_choice_is_not_overwritten(
    monkeypatch, tmp_path, one_roi_gdf, captured_downloads
):
    """A user asking for VV only must not have the VV+VH default put back."""
    download_one_roi(
        monkeypatch,
        one_roi_gdf,
        tmp_path,
        sentinel_1_properties={
            "transmitterReceiverPolarisation": ["VV"],
            "instrumentMode": "IW",
        },
    )

    inputs = captured_downloads[0]
    assert inputs["sentinel_1_properties"]["transmitterReceiverPolarisation"] == ["VV"]


# ---------------------------------------------------------------------------
# An incomplete site reaches coastsat intact
# ---------------------------------------------------------------------------


def test_a_vh_only_site_is_handed_straight_to_coastsat(
    sar_legacy_site_with_meta, captured_downloads
):
    """A site missing a polarization must not be intercepted on the way to the download.

    CoastSeg used to warn here first. It no longer does: coastsat re-queues the
    incomplete scenes and backfills them in place during this very call, and reports
    that itself in terms of scenes rather than polarization folders. Anything CoastSeg
    printed beforehand would have told the user to re-run the download they had already
    started.
    """
    inputs = sar_legacy_site_with_meta["inputs"]

    coastseg_map.CoastSeg_Map(create_map=False).retrieve_images_for_rois([inputs], {})

    assert len(captured_downloads) == 1
    assert captured_downloads[0] is inputs


def test_the_incomplete_site_is_left_on_disk_untouched(
    sar_legacy_site_with_meta, captured_downloads
):
    """Nothing on the way to the download may delete the bands already downloaded.

    The backfill overwrites scenes in place under the same sitename, so the VH already
    on disk has to still be there when coastsat takes over.
    """
    coastseg_map.CoastSeg_Map(create_map=False).retrieve_images_for_rois(
        [sar_legacy_site_with_meta["inputs"]], {}
    )

    assert os.path.isfile(sar_legacy_site_with_meta["vh"])
    assert os.path.isfile(sar_legacy_site_with_meta["meta"])


# ---------------------------------------------------------------------------
# Resuming a VH-only download
# ---------------------------------------------------------------------------


def test_a_vh_only_scene_is_re_queued_for_download(sar_legacy_site_with_meta):
    """coastsat must not report the scene as already downloaded just because the date is
    on disk -- that is what left legacy sites single-polarization forever."""
    inputs = sar_legacy_site_with_meta["inputs"]
    metadata = es.get_metadata(inputs)

    assert metadata["S1"]["band_order"] == [["VH"]]

    incomplete = SDS_download.get_incomplete_sar_dates(
        inputs, metadata["S1"], ["VV", "VH"]
    )

    assert len(incomplete) == 1


def test_backfilling_vv_clears_the_incomplete_scene(sar_legacy_site_with_meta):
    """Once VV lands next to VH the scene is complete and must not be fetched again."""
    inputs = sar_legacy_site_with_meta["inputs"]
    sar_legacy_site_with_meta["add_vv"]()
    metadata = es.get_metadata(inputs)

    incomplete = SDS_download.get_incomplete_sar_dates(
        inputs, metadata["S1"], ["VV", "VH"]
    )

    assert incomplete == set()


def test_the_resumed_site_keeps_its_sitename_and_gains_no_duplicate(
    sar_legacy_site_with_meta,
):
    """The backfill overwrites the scene in place: one file per polarization, no _dup
    suffix, and both polarizations present."""
    sar_legacy_site_with_meta["add_vv"]()

    s1_root = os.path.join(sar_legacy_site_with_meta["site"], "S1")
    for polarization in ("VV", "VH"):
        tifs = [
            name
            for name in os.listdir(os.path.join(s1_root, polarization))
            if name.endswith(".tif")
        ]
        assert len(tifs) == 1
        assert "_dup" not in tifs[0]
        assert tifs[0].startswith(sar_legacy_site_with_meta["scene_date"])
