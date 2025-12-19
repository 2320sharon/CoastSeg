import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from coastseg.roi import ROI


def _new_roi() -> ROI:
    """Create an ROI instance without running its __init__ for gdf property tests."""
    return ROI.__new__(ROI)


def test_gdf_lazy_creation_when_missing():
    roi = _new_roi()

    result = roi.gdf

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty
    assert result.crs.to_string() == ROI.DEFAULT_CRS


def test_gdf_sets_default_when_none_assigned():
    roi = _new_roi()

    roi.gdf = None

    assert isinstance(roi.gdf, gpd.GeoDataFrame)
    assert roi.gdf.empty
    assert roi.gdf.crs.to_string() == ROI.DEFAULT_CRS


def test_gdf_preserves_existing_non_default_crs():
    roi = _new_roi()
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:3857"
    )

    roi.gdf = gdf

    assert roi.gdf.crs.to_string() == "EPSG:3857"
    assert len(roi.gdf) == 1


def test_gdf_accepts_default_crs():
    roi = _new_roi()
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs=ROI.DEFAULT_CRS
    )

    roi.gdf = gdf

    assert roi.gdf.crs.to_string() == ROI.DEFAULT_CRS
    assert len(roi.gdf) == 1


def test_gdf_sets_default_for_empty_gdf_without_crs():
    roi = _new_roi()
    gdf = gpd.GeoDataFrame()

    roi.gdf = gdf

    assert roi.gdf.crs.to_string() == ROI.DEFAULT_CRS
    assert roi.gdf.empty
