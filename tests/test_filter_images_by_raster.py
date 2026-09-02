"""Tests for measuring a scene's footprint from its GeoTIFF rather than its preview jpg.

`filter_images` used to infer ground area from the preview's pixel dimensions. That holds
for optical, whose preview is a raw raster write, but coastsat renders the Sentinel-1
preview as a titled matplotlib figure -- figsize x dpi, plus axes and whitespace. Measuring
that made every S1 scene look several times larger than its ROI, so all of them were
rejected as oversized and moved to `bad`.
"""

import os

import numpy as np
import pytest
from PIL import Image

from coastseg import common


PIXEL_SIZE = {"S1": 10, "S2": 10, "L8": 15}


def write_tif(path, width, height, pixel_size, epsg=32610):
    """A minimal georeferenced GeoTIFF of the given size."""
    from osgeo import gdal, osr

    gdal.UseExceptions()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = gdal.GetDriverByName("GTiff").Create(path, width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((500000.0, pixel_size, 0.0, 4000000.0, 0.0, -pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.ones((height, width), dtype=np.float32))
    dataset = None
    return path


def write_jpg(path, width, height):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (width, height), (128, 128, 128)).save(path)
    return path


@pytest.fixture
def s1_site(tmp_path):
    """A 448x448 S1 scene (20.07 km2) whose preview is a much larger figure."""
    site = tmp_path / "ID_test_datetime01-01-26__00_00_00"
    date = "2023-05-06-02-07-48"
    write_tif(str(site / "S1" / "VV" / f"{date}_S1_site_VV.tif"), 448, 448, 10)
    write_tif(str(site / "S1" / "VH" / f"{date}_S1_site_VH.tif"), 448, 448, 10)
    rgb = site / "jpg_files" / "preprocessed" / "RGB"
    write_jpg(str(rgb / f"{date}_RGB_S1.jpg"), 962, 988)  # the matplotlib figure
    return {"site": str(site), "rgb": str(rgb), "date": date}


# ---------------------------------------------------------------------------
# calculate_raster_area
# ---------------------------------------------------------------------------


def test_area_comes_from_the_geotransform_not_a_hardcoded_table(tmp_path):
    path = write_tif(str(tmp_path / "scene.tif"), 448, 448, 10)

    # the fallback pixel size is deliberately wrong; the geotransform must win
    assert common.calculate_raster_area(path, pixel_size=999) == pytest.approx(20.0704)


def test_a_missing_raster_is_zero_rather_than_an_exception(tmp_path):
    assert common.calculate_raster_area(str(tmp_path / "nope.tif")) == 0.0


def test_landsat_pixel_size_is_read_correctly(tmp_path):
    path = write_tif(str(tmp_path / "l8.tif"), 300, 300, 15)

    assert common.calculate_raster_area(path) == pytest.approx(20.25)


# ---------------------------------------------------------------------------
# find_source_raster
# ---------------------------------------------------------------------------


def test_an_s1_preview_resolves_to_a_polarization_tif(s1_site):
    found = common.find_source_raster(f"{s1_site['date']}_RGB_S1.jpg", s1_site["site"])

    assert found is not None
    assert found.endswith(".tif")
    assert os.path.join("S1", "VV") in found  # canonical order, VV first


def test_an_s1_preview_still_resolves_when_only_vh_exists(tmp_path):
    """A legacy VH-only site must not lose its size filter."""
    site = tmp_path / "site"
    date = "2023-05-06-02-07-48"
    write_tif(str(site / "S1" / "VH" / f"{date}_S1_site_VH.tif"), 448, 448, 10)

    found = common.find_source_raster(f"{date}_RGB_S1.jpg", str(site))

    assert found is not None and os.path.join("S1", "VH") in found


def test_an_optical_preview_resolves_to_the_ms_tif(tmp_path):
    site = tmp_path / "site"
    date = "2023-06-16-15-53-12"
    write_tif(str(site / "S2" / "ms" / f"{date}_S2_site_ms.tif"), 456, 456, 10)

    found = common.find_source_raster(f"{date}_RGB_S2.jpg", str(site))

    assert found is not None and os.path.join("S2", "ms") in found


def test_no_raster_resolves_to_none(tmp_path):
    assert common.find_source_raster("2023-06-16-15-53-12_RGB_S2.jpg", str(tmp_path)) is None
    assert common.find_source_raster("2023-06-16-15-53-12_RGB_S2.jpg", "") is None


# ---------------------------------------------------------------------------
# filter_images
# ---------------------------------------------------------------------------


def test_an_s1_scene_matching_its_roi_is_kept(s1_site):
    """The regression: a 20.07 km2 scene inside a 20 km2 ROI's 8-30 km2 window.

    Measured from the 962x988 preview it comes out at 95 km2 and is rejected.
    """
    bad = common.filter_images(8.0, 30.0, s1_site["rgb"], roi_directory=s1_site["site"])

    assert bad == []
    assert os.path.exists(os.path.join(s1_site["rgb"], f"{s1_site['date']}_RGB_S1.jpg"))


def test_the_preview_alone_would_have_rejected_it(s1_site):
    """Pins why the fix is needed: without the GeoTIFF the same scene fails."""
    bad = common.filter_images(8.0, 30.0, s1_site["rgb"])  # no roi_directory

    # S1 is skipped rather than judged on a meaningless number
    assert bad == []
    jpg = os.path.join(s1_site["rgb"], f"{s1_site['date']}_RGB_S1.jpg")
    assert common.calculate_image_area(jpg, 10) > 30.0  # what the old code measured


def test_a_genuinely_partial_s1_scene_is_still_rejected(s1_site):
    """The filter must keep doing its job, not just pass everything."""
    date = "2023-05-18-02-07-49"
    write_tif(str(os.path.join(s1_site["site"], "S1", "VV", f"{date}_S1_site_VV.tif")),
              448, 100, 10)  # ~4.5 km2, well under the 8 km2 floor
    write_jpg(os.path.join(s1_site["rgb"], f"{date}_RGB_S1.jpg"), 962, 988)

    bad = common.filter_images(8.0, 30.0, s1_site["rgb"], roi_directory=s1_site["site"])

    assert [os.path.basename(f) for f in bad] == [f"{date}_RGB_S1.jpg"]


def test_optical_results_are_unchanged_by_measuring_the_raster(tmp_path):
    """Optical previews are raw raster writes, so both routes must agree."""
    site = tmp_path / "site"
    rgb = site / "jpg_files" / "preprocessed" / "RGB"
    date = "2017-12-09-16-57-21"
    write_tif(str(site / "L8" / "ms" / f"{date}_L8_site_ms.tif"), 300, 300, 15)
    jpg = write_jpg(str(rgb / f"{date}_RGB_L8.jpg"), 300, 300)

    from_raster = common.calculate_raster_area(
        common.find_source_raster(os.path.basename(jpg), str(site))
    )

    assert from_raster == pytest.approx(common.calculate_image_area(jpg, 15))
    assert common.filter_images(8.0, 30.0, str(rgb), roi_directory=str(site)) == []


def test_bad_images_are_moved_into_the_bad_folder(s1_site):
    date = "2023-05-18-02-07-49"
    write_tif(str(os.path.join(s1_site["site"], "S1", "VV", f"{date}_S1_site_VV.tif")),
              448, 100, 10)
    write_jpg(os.path.join(s1_site["rgb"], f"{date}_RGB_S1.jpg"), 962, 988)

    common.filter_images(8.0, 30.0, s1_site["rgb"], roi_directory=s1_site["site"])

    assert os.path.isfile(os.path.join(s1_site["rgb"], "bad", f"{date}_RGB_S1.jpg"))
    assert not os.path.isfile(os.path.join(s1_site["rgb"], f"{date}_RGB_S1.jpg"))
