import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from coastseg import exceptions
from coastseg.shoreline import Shoreline


def test_shoreline_initialization():
    shoreline = Shoreline()
    assert isinstance(shoreline, Shoreline)
    assert isinstance(shoreline.gdf, gpd.GeoDataFrame)


# Test that when shoreline is not a linestring an error is thrown
def test_shoreline_wrong_geometry():
    polygon_gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-122.66944064253451, 36.96768728778939),
                    (-122.66944064253451, 34.10377172691159),
                    (-117.75040020737816, 34.10377172691159),
                    (-117.75040020737816, 36.96768728778939),
                    (-122.66944064253451, 36.96768728778939),
                ]
            )
        ],
        crs="epsg:4326",
    )
    with pytest.raises(exceptions.InvalidGeometryType):
        Shoreline(shoreline=polygon_gdf)


# 1. load shorelines from a shorelines geodataframe with a CRS 4326 with no id
def test_initialize_shorelines_with_provided_shorelines(valid_shoreline_gdf):
    actual_shoreline = Shoreline(shoreline=valid_shoreline_gdf)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


# 2. load shorelines from a shorelines geodataframe with a CRS 4327 with no id
def test_initialize_shorelines_with_wrong_CRS(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4327", inplace=False)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


def test_intersecting_files(box_no_shorelines_transects):
    """
    Test case to verify the behavior of get_intersecting_shoreline_files
    when there are no intersecting shoreline files.

    Args:
        box_no_shorelines_transects: The box with no default shoreline or transects.

    """
    sl = Shoreline()
    assert sl.get_intersecting_shoreline_files(box_no_shorelines_transects) == []


def test_intersecting_files_valid_bbox(valid_bbox_gdf):
    """
    Test case to check if the get_intersecting_shoreline_files method returns a non-empty list
    when provided with a valid bounding box GeoDataFrame.
    """
    sl = Shoreline()
    assert sl.get_intersecting_shoreline_files(valid_bbox_gdf) != []


# 3. load shorelines from a shorelines geodataframe with empty ids
def test_initialize_shorelines_with_empty_id_column(valid_shoreline_gdf):
    """
    Test case to verify the initialization of shorelines with an empty 'id' column.

    Args:
        valid_shoreline_gdf (geopandas.GeoDataFrame): A valid GeoDataFrame containing shoreline data.
    """
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4326", inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id=None)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


# 4. load shorelines from a shorelines geodataframe with identical ids
def test_initialize_shorelines_with_identical_ids(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4326", inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id="bad_id")
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


def test_initialize_shorelines_with_bbox(valid_bbox_gdf):
    shoreline = Shoreline(bbox=valid_bbox_gdf)

    assert not shoreline.gdf.empty
    assert "id" in shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(shoreline.gdf["id"].duplicated()) == True


def test_style_layer():
    layer_name = "test_layer"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [125.6, 10.1]},
                "properties": {"name": "test"},
            }
        ],
    }
    shoreline = Shoreline()
    layer = shoreline.style_layer(geojson_data, layer_name)

    assert layer.name == layer_name
    assert (
        layer.data["features"][0]["geometry"] == geojson_data["features"][0]["geometry"]
    )
    assert layer.style


# # you can also mock some methods that depend on external data, like downloading from the internet
# this requires the use of the pytest-mock library which must be installed ( it is not as of 2/8/2024)
# def test_download_shoreline(mocker):
#     mock_download = mocker.patch("coastseg.common.download_url", return_value=None)
#     shoreline = Shoreline()
#     with exceptions.DownloadError:
#         shoreline.download_shoreline("test_file.geojson")

#     mock_download.assert_called_once_with("https://zenodo.org/record/7761607/files/test_file.geojson?download=1", mocker.ANY, filename="test_file.geojson")
