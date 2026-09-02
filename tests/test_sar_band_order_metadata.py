"""Tests for reading Sentinel-1 `band_order` out of a scene's metadata .txt file.

coastsat writes one metadata file per S1 scene carrying the polarizations it downloaded.
CoastSeg keeps its own copy of `get_metadata`, which used to drop that key entirely.
coastsat reads it back when it cannot list the files on disk -- for example to tell a
complete scene from one still missing a band -- so it has to survive the round trip and
stay index-aligned with `filenames`.
"""

import os

import pytest

from coastseg import extracted_shoreline as es


LEGACY_META = """\
filename\t2024-08-02-08-39-41_S1_site_VH.tif
epsg\t32756
im_width\t168
im_height\t436
orbitProperties_pass\tASCENDING
transmitterReceiverPolarisation\t['VV', 'VH']
saved_polarization\tVH
instrumentMode\tIW
"""

DUAL_POL_META = """\
filename\t2023-12-03-19-15-49_S1_site_VV.tif
epsg\t32756
im_width\t168
im_height\t436
band_order\t['VV', 'VH']
orbitProperties_pass\tDESCENDING
transmitterReceiverPolarisation\t['VV', 'VH']
instrumentMode\tIW
"""


def write_meta(directory, filename, contents):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(contents)
    return path


@pytest.fixture
def s1_site(tmp_path):
    """A site whose meta folder the caller fills in."""
    meta_dir = tmp_path / "site" / "S1" / "meta"
    meta_dir.mkdir(parents=True)
    return tmp_path, meta_dir


def read_s1_metadata(root, dates=("2020-01-01", "2030-01-01")):
    return es.get_metadata(
        {
            "filepath": str(root),
            "sitename": "site",
            "sat_list": ["S1"],
            "dates": list(dates),
        }
    )


# ---------------------------------------------------------------------------
# read_metadata_file
# ---------------------------------------------------------------------------


def test_band_order_is_parsed_into_a_list(tmp_path):
    path = write_meta(str(tmp_path), "scene.txt", DUAL_POL_META)

    assert es.read_metadata_file(path)["band_order"] == ["VV", "VH"]


def test_a_legacy_file_has_no_band_order(tmp_path):
    path = write_meta(str(tmp_path), "scene.txt", LEGACY_META)
    meta_info = es.read_metadata_file(path)

    assert meta_info["band_order"] == []
    assert meta_info["saved_polarization"] == "VH"


def test_a_malformed_band_order_does_not_break_the_read(tmp_path, caplog):
    """band_order is optional; a bad value must not take down the whole file."""
    path = write_meta(
        str(tmp_path), "scene.txt", DUAL_POL_META.replace("['VV', 'VH']", "[VV, VH")
    )

    with caplog.at_level("WARNING"):
        meta_info = es.read_metadata_file(path)

    assert meta_info["band_order"] == []
    assert meta_info["epsg"] == 32756  # the rest of the file still parsed
    assert "band_order" in caplog.text


# ---------------------------------------------------------------------------
# get_band_order_from_meta
# ---------------------------------------------------------------------------


def test_band_order_wins_over_saved_polarization():
    resolved = es.get_band_order_from_meta(
        {"band_order": ["VV", "VH"], "saved_polarization": "VH"}
    )
    assert resolved == ["VV", "VH"]


def test_a_legacy_scene_falls_back_to_saved_polarization():
    assert es.get_band_order_from_meta({"saved_polarization": "VH"}) == ["VH"]


def test_neither_key_resolves_to_empty():
    assert es.get_band_order_from_meta({"filename": "x.tif"}) == []


def test_the_requested_polarizations_are_not_mistaken_for_the_saved_ones():
    """transmitterReceiverPolarisation is what was asked for, not what landed on disk --
    a legacy file carries ['VV','VH'] there while holding only VH."""
    resolved = es.get_band_order_from_meta(
        {"transmitterReceiverPolarisation": ["VV", "VH"], "saved_polarization": "VH"}
    )
    assert resolved == ["VH"]


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------


def test_get_metadata_records_band_order_for_s1(s1_site):
    root, meta_dir = s1_site
    write_meta(str(meta_dir), "2023-12-03-19-15-49_S1_site.txt", DUAL_POL_META)

    metadata = read_s1_metadata(root)

    assert metadata["S1"]["band_order"] == [["VV", "VH"]]


def test_band_order_stays_index_aligned_on_a_mixed_site(s1_site):
    """A legacy VH-only scene sitting next to a new dual-pol one. band_order[i] must
    stay safe to index alongside filenames[i]."""
    root, meta_dir = s1_site
    write_meta(str(meta_dir), "2023-12-03-19-15-49_S1_site.txt", DUAL_POL_META)
    write_meta(str(meta_dir), "2024-08-02-08-39-41_S1_site_VH.txt", LEGACY_META)

    s1 = read_s1_metadata(root)["S1"]

    assert len(s1["band_order"]) == len(s1["filenames"]) == 2
    assert s1["band_order"] == [["VV", "VH"], ["VH"]]


def test_optical_satellites_do_not_gain_a_band_order_key(tmp_path):
    meta_dir = tmp_path / "site" / "L8" / "meta"
    meta_dir.mkdir(parents=True)
    write_meta(
        str(meta_dir),
        "2023-12-03-19-15-49_L8_site.txt",
        "filename\t2023-12-03-19-15-49_L8_site_ms.tif\nepsg\t32756\n",
    )

    metadata = es.get_metadata(
        {
            "filepath": str(tmp_path),
            "sitename": "site",
            "sat_list": ["L8"],
            "dates": ["2020-01-01", "2030-01-01"],
        }
    )

    assert "band_order" not in metadata["L8"]
