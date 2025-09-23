# Standard library
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Union

# Third-party
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

# Project / local
from coastseg.common import (
    create_unique_ids,
    preprocess_geodataframe,
    validate_geometry_types,
)
from coastseg.feature import Feature

logger = logging.getLogger(__name__)


def drop_columns(
    gdf: gpd.GeoDataFrame, columns_to_drop: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Drop specified columns from a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame.
        columns_to_drop (Optional[List[str]], optional): Column names to drop.
            If None, drops default oceanographic columns.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with specified columns removed.
    """
    if columns_to_drop is None:
        columns_to_drop = [
            "MEAN_SIG_WAVEHEIGHT",
            "TIDAL_RANGE",
            "ERODIBILITY",
            "river_label",
            "sinuosity_label",
            "slope_label",
            "turbid_label",
        ]
    for col in columns_to_drop:
        if col in gdf.columns:
            gdf.drop(columns=[col], inplace=True)
    return gdf


def create_transects_with_arrowheads(
    gdf: gpd.GeoDataFrame, arrow_length: float = 0.0004, arrow_angle: int = 30
) -> gpd.GeoDataFrame:
    """
    Creates transects with arrowheads to indicate direction.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing transects with LineString geometries.
        arrow_length (float, optional): Length of arrowhead in CRS units. Defaults to 0.0004.
        arrow_angle (int, optional): Angle of arrowhead in degrees. Defaults to 30.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with merged geometries of transects and arrowheads.
    """
    gdf_copy = gdf.to_crs("EPSG:4326")
    # remove unneeded columns
    gdf_copy = drop_columns(gdf_copy)

    # Create arrowheads for each transect
    gdf_copy["arrowheads"] = gdf_copy["geometry"].apply(
        lambda x: create_arrowhead(
            x, arrow_length=arrow_length, arrow_angle=arrow_angle
        )  # type: ignore
    )
    # Merge each transect with its arrowhead
    gdf_copy["merged"] = gdf_copy.apply(
        lambda row: unary_union([row["geometry"], row["arrowheads"]]), axis=1
    )
    gdf_copy.rename(
        columns={"geometry": "transect_geometry", "merged": "geometry"}, inplace=True
    )
    if "arrowheads" in gdf_copy.columns:
        gdf_copy.drop(columns=["arrowheads"], inplace=True)
    if "transect_geometry" in gdf_copy.columns:
        gdf_copy.drop(columns=["transect_geometry"], inplace=True)

    return gdf_copy


# Function to create an arrowhead as a triangle polygon this works in crs 4326
def create_arrowhead(
    line: LineString, arrow_length: float = 0.0004, arrow_angle: float = 30
) -> Polygon:
    """
    Create an arrowhead polygon at the end of a line.

    Args:
        line (LineString): Line geometry to create arrowhead for.
        arrow_length (float, optional): Arrowhead length in CRS units. Defaults to 0.0004 for CRS EPSG:4326.
        arrow_angle (float, optional): Arrowhead angle in degrees. Defaults to 30.

    Returns:
        Polygon: Triangular arrowhead polygon positioned at line end.
    """
    # Get the last segment of the line
    p1, p2 = line.coords[-2], line.coords[-1]
    # Calculate the angle of the line
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    # Calculate the points of the arrowhead
    arrow_angle_rad = math.radians(arrow_angle)
    left_angle = angle - math.pi + arrow_angle_rad
    right_angle = angle - math.pi - arrow_angle_rad

    left_point = (
        p2[0] + arrow_length * math.cos(left_angle),
        p2[1] + arrow_length * math.sin(left_angle),
    )
    right_point = (
        p2[0] + arrow_length * math.cos(right_angle),
        p2[1] + arrow_length * math.sin(right_angle),
    )

    return Polygon([p2, left_point, right_point])


def load_intersecting_transects(
    rectangle: gpd.GeoDataFrame,
    transect_files: List[str],
    transect_dir: str,
    columns_to_keep: Iterable[str] = {"id", "geometry", "slope"},
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Load transects that intersect with a rectangular bounding box.

    Args:
        rectangle (gpd.GeoDataFrame): GeoDataFrame containing rectangular geometry for filtering.
        transect_files (List[str]): List of GeoJSON filenames to read.
        transect_dir (str): Directory path containing GeoJSON transect files.
        columns_to_keep (Iterable[str], optional): Column names to retain.
            Defaults to {"id", "geometry", "slope"}.
        **kwargs: Additional keyword arguments.

    Keyword Args:
        crs (str, optional): Target CRS. Defaults to "EPSG:4326".

    Returns:
        gpd.GeoDataFrame: Transects that intersect with rectangle, with specified columns retained.

    Raises:
        FileNotFoundError: If transect file does not exist.
        ValueError: If no valid transects found.
    """
    crs = kwargs.get("crs", "EPSG:4326")

    # Create an empty GeoDataFrame to hold the selected transects
    selected_transects = gpd.GeoDataFrame(columns=list(columns_to_keep), crs=crs)

    # Get the bounding box of the rectangle in the same CRS as the transects
    if hasattr(rectangle, "crs") and rectangle.crs:
        rectangle = rectangle.copy().to_crs(crs)
    else:
        rectangle = rectangle.copy().set_crs(crs)
    # get the bounding box of the rectangle
    bbox = tuple(rectangle.bounds.iloc[0].tolist())
    # Create a list to store the GeoDataFrames
    gdf_list = []
    # Iterate over each transect file and select the transects that intersect with the rectangle
    for transect_file in transect_files:
        transects_name = os.path.splitext(transect_file)[0]
        transect_path = os.path.join(transect_dir, transect_file)
        if not os.path.exists(transect_path):
            logger.warning("Transect file %s does not exist", transect_path)
            continue
        transects = gpd.read_file(transect_path, bbox=bbox)
        # keep only those transects that intersect with the rectangle
        transects = transects[transects.intersects(rectangle.unary_union)]
        # drop any columns that are not in columns_to_keep
        columns_to_keep = set(col.lower() for col in columns_to_keep)
        transects = transects[
            [col for col in transects.columns if col.lower() in columns_to_keep]
        ]
        # if the transects are not empty then add them to the list
        if not transects.empty:
            logger.info("Adding transects from %s", transects_name)
            gdf_list.append(transects)

    # Concatenate all the GeoDataFrames in the list into one GeoDataFrame
    if gdf_list:
        selected_transects = pd.concat(gdf_list, ignore_index=True)

    if not selected_transects.empty:
        selected_transects = preprocess_geodataframe(
            selected_transects,
            columns_to_keep=list(columns_to_keep),
            create_ids=True,
            output_crs=crs,
        )
    # ensure that the transects are either LineStrings or MultiLineStrings
    validate_geometry_types(
        selected_transects,
        set(["LineString", "MultiLineString"]),
        feature_type="transects",
    )
    # make sure all the ids in selected_transects are unique
    selected_transects = create_unique_ids(selected_transects, prefix_length=3)
    return selected_transects


class Transects(Feature):
    """A class representing a collection of transects within a specified bounding box."""

    LAYER_NAME = "transects"
    COLUMNS_TO_KEEP = set(
        [
            "id",
            "geometry",
            "slope",
            "distance",
            "feature_x",
            "feature_y",
            "nearest_x",
            "nearest_y",
        ]
    )

    # COLUMNS_TO_KEEP
    # ---------------
    # id: unique identifier for each transect
    # geometry: the geometric shape, position, and configuration of the transect
    # slope: represents the beach face slope, used for tidal correction of transect-based data
    # distance: represents the distance in degrees between the slope datum location and the transect location
    # feature_x: x-coordinate of the transect location
    # feature_y: y-coordinate of the transect location
    # nearest_x: x-coordinate of the nearest slope location to the transect
    # nearest_y: y-coordinate of the nearest slope location to the transect

    def __init__(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        transects: Optional[gpd.GeoDataFrame] = None,
        filename: Optional[str] = None,
    ):
        """
        Initialize a Transects object for coastal transect analysis.

        Args:
            bbox (Optional[gpd.GeoDataFrame], optional): Bounding box geometry to filter transects.
                Defaults to None.
            transects (Optional[gpd.GeoDataFrame], optional): Existing transect data.
                Defaults to None.
            filename (Optional[str], optional): Filename for saving/loading transect data.
                Defaults to "transects.geojson".

        Note:
            Only one of `bbox` or `transects` should be provided. If both are provided,
            `transects` takes precedence.
        """
        self.gdf = gpd.GeoDataFrame()
        self.filename = filename if filename else "transects.geojson"
        self.initialize_transects(bbox, transects)

    def __str__(self) -> str:
        """
        Return a string representation of the Transects object.

        Returns:
            str: Formatted string containing transects summary with CRS, column types,
                data preview, and transect IDs.
        """
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf.head().to_string()
        # Get CRS information
        if self.gdf.empty:
            crs_info = "CRS: None"
        else:
            if self.gdf is not None and hasattr(self.gdf, "crs"):
                crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
            else:
                crs_info = "CRS: None"
        ids = ""
        if "id" in self.gdf.columns:
            ids = self.gdf["id"].astype(str)
        return f"Transects:\nself.gdf:\n{crs_info}\n- Columns and Data Types:\n{col_info}\n\n- First 5 Rows:\n{first_rows}\nIDs:\n{ids}"

    def __repr__(self) -> str:
        """
        Return a string representation of the Transects object for debugging.

        Returns:
            str: Formatted string containing detailed transects information for debugging.
        """
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf.head().to_string()
        # Get CRS information
        if self.gdf.empty:
            crs_info = "CRS: None"
        else:
            if self.gdf is not None and hasattr(self.gdf, "crs"):
                crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
            else:
                crs_info = "CRS: None"
        ids = ""
        if "id" in self.gdf.columns:
            ids = self.gdf["id"].astype(str)
        return f"Transects:\nself.gdf:\n{crs_info}\n- Columns and Data Types:\n{col_info}\n\n- First 5 Rows:\n{first_rows}\nIDs:\n{ids}"

    def initialize_transects(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        transects: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """
        Initialize the transects data using either a bounding box or existing transects.

        Args:
            bbox (Optional[gpd.GeoDataFrame], optional): Bounding box geometry to filter transects.
                Defaults to None.
            transects (Optional[gpd.GeoDataFrame], optional): Existing transect data.
                Defaults to None.

        Note:
            If both parameters are provided, `transects` takes precedence over `bbox`.
        """
        if transects is not None:
            self.initialize_transects_with_transects(transects)

        elif bbox is not None:
            self.initialize_transects_with_bbox(bbox)

    def initialize_transects_with_transects(self, transects: gpd.GeoDataFrame) -> None:
        """
        Initialize the transects object with existing transect data.

        Args:
            transects (gpd.GeoDataFrame): Existing transect data with LineString or
                MultiLineString geometries.

        Raises:
            ValueError: If transects contain invalid geometry types.
        """
        if not transects.empty:
            if not transects.crs:
                logger.warning(
                    f"transects did not have a crs converting to crs 4326 \n {transects}"
                )
                transects.set_crs("EPSG:4326", inplace=True)
            transects = preprocess_geodataframe(
                transects,
                columns_to_keep=list(Transects.COLUMNS_TO_KEEP),
                create_ids=True,
                output_crs="EPSG:4326",
            )
            validate_geometry_types(
                transects,
                set(["LineString", "MultiLineString"]),
                feature_type="transects",
                help_message="The uploaded transects need to be LineStrings.",
            )
            # if not all the ids in transects are unique then create unique ids
            transects = create_unique_ids(transects, prefix_length=3)
            # if an id column exists then make sure it is a string
            if "id" in transects.columns:
                transects["id"] = transects["id"].astype(str)
            self.gdf = transects

    def initialize_transects_with_bbox(self, bbox: gpd.GeoDataFrame) -> None:
        """
        Initialize transects by loading those that intersect with a bounding box.

        Args:
            bbox (gpd.GeoDataFrame): Bounding box geometry to filter transects.
        """
        if not bbox.empty:
            self.gdf = self.create_geodataframe(bbox)

    def create_geodataframe(
        self,
        bbox: gpd.GeoDataFrame,
        crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame of transects that intersect with a bounding box.

        Searches for transect files in the data directory, identifies those
        that spatially intersect with the provided bounding box, and loads them into
        a GeoDataFrame. The resulting transects are validated, and converted to the
        specified coordinate reference system.

        Args:
            bbox (gpd.GeoDataFrame): Bounding box geometry to filter transects.
            crs (str, optional): Target CRS for output transects. Defaults to "EPSG:4326".

        Returns:
            gpd.GeoDataFrame: Transects that intersect with the bounding box.
        """
        # create a new dataframe that only contains the geometry column of the bbox
        bbox = bbox[["geometry"]]
        # get transect geojson files that intersect with bounding box
        intersecting_transect_files = self.get_intersecting_files(bbox)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        transect_dir = os.path.abspath(os.path.join(script_dir, "transects"))
        # for each transect file clip it to the bbox and add to map
        transects_in_bbox = load_intersecting_transects(
            bbox,
            intersecting_transect_files,
            transect_dir,
            columns_to_keep=list(Transects.COLUMNS_TO_KEEP),
        )
        if transects_in_bbox.empty:
            logger.warning("No transects found here.")
            return transects_in_bbox
        # remove z-axis from transects
        transects_in_bbox = preprocess_geodataframe(
            transects_in_bbox,
            columns_to_keep=list(Transects.COLUMNS_TO_KEEP),
            create_ids=True,
        )
        validate_geometry_types(
            transects_in_bbox,
            set(["LineString", "MultiLineString"]),
            feature_type="transects",
        )
        # make sure all the ids in transects_in_bbox are unique
        transects_in_bbox = create_unique_ids(transects_in_bbox, prefix_length=3)

        if not transects_in_bbox.empty:
            transects_in_bbox.to_crs(crs, inplace=True)

        return transects_in_bbox

    def style_layer(
        self,
        data: Union[Dict[str, Any], gpd.GeoDataFrame],
        layer_name: str = "transects",
    ) -> Dict[str, Any]:
        """
        Create a styled GeoJSON layer for transects visualization.

        Args:
            data (Union[Dict[str, Any], gpd.GeoDataFrame]): Transect data to style.
            layer_name (str, optional): Layer name for visualization. Defaults to "transects".

        Returns:
            Dict[str, Any]: Styled GeoJSON layer for interactive mapping.
        """
        geojson = data
        if isinstance(data, dict):
            geojson = data
        elif isinstance(data, gpd.GeoDataFrame):
            gdf = create_transects_with_arrowheads(data, arrow_angle=30)
            geojson = json.loads(gdf.to_json())

        style = {
            "color": "grey",
            "fill_color": "grey",
            "opacity": 1,
            "fillOpacity": 0.2,
            "weight": 2,
        }
        hover_style = {"color": "blue", "fillOpacity": 0.7}
        return super().style_layer(
            geojson, layer_name, style=style, hover_style=hover_style
        )

    def get_intersecting_files(self, bbox_gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Get transect filenames that spatially intersect with the bounding box.

        Args:
            bbox_gdf (gpd.GeoDataFrame): Bounding box geometry to test for intersection.

        Returns:
            List[str]: Transect filenames that intersect with the bounding box.
        """
        # dataframe containing total bounding box for each transects file
        total_bounds_df = self.load_total_bounds_df()
        # filenames where transects/shoreline's bbox intersect bounding box drawn by user
        intersecting_files = []
        for filename in total_bounds_df.index:
            minx, miny, maxx, maxy = total_bounds_df.loc[filename]
            intersection_df = bbox_gdf.cx[minx:maxx, miny:maxy]
            # save filenames where gpd_bbox & bounding box for set of transects intersect
            if not intersection_df.empty:
                intersecting_files.append(filename)
        return intersecting_files

    def load_total_bounds_df(self) -> pd.DataFrame:
        """
        Load DataFrame containing bounding box information for all transect files.

        Returns:
            pd.DataFrame: DataFrame with filenames as index and bounding box coordinates
                (minx, miny, maxx, maxy) for each transect file.

        Raises:
            FileNotFoundError: If bounding boxes CSV file is not found.
        """
        # Load in the total bounding box from csv
        # Create the directory to hold the downloaded shorelines from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bounding_box_dir = os.path.abspath(os.path.join(script_dir, "bounding_boxes"))
        if not os.path.exists(bounding_box_dir):
            os.mkdir(bounding_box_dir)

        transects_csv = os.path.join(bounding_box_dir, "transects_bounding_boxes.csv")
        if not os.path.exists(transects_csv):
            print("Did not find transects csv at ", transects_csv)
            return pd.DataFrame()  # Return empty DataFrame if file not found
        else:
            total_bounds_df = pd.read_csv(transects_csv)

        # Set filename as index for efficient lookup
        total_bounds_df = total_bounds_df.set_index("filename")
        return total_bounds_df
