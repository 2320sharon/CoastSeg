# Standard library imports
import datetime
import logging
from collections.abc import Iterable
from typing import Any, Collection, Dict, List, Optional, Union, cast

# External dependencies imports
import geopandas as gpd
import pandas as pd
from ipyleaflet import GeoJSON
from shapely import geometry

# Internal dependencies imports
from coastseg import common, exceptions
from coastseg.feature import Feature

logger = logging.getLogger(__name__)

__all__ = ["ROI"]


def get_ids_with_invalid_area(
    geometry: gpd.GeoDataFrame, max_area: float = 98000000, min_area: float = 0
) -> List[str]:
    """
    Get indices of geometries with areas outside the specified range.

    Assumes GeoDataFrame is in EPSG:4326. Areas returned in meters squared.

    Args:
        geometry: GeoDataFrame with geometries to check.
        max_area: Maximum allowable area.
        min_area: Minimum allowable area.

    Returns:
        Set of indices for invalid geometries.

    Raises:
        TypeError: If geometry is not a GeoDataFrame.
    """
    if not isinstance(geometry, gpd.GeoDataFrame):
        raise TypeError("Must be GeoDataFrame")

    # Project to UTM for accurate area calculation
    projected = geometry.to_crs(geometry.estimate_utm_crs())
    areas = projected.area

    # Get indices where area is invalid
    invalid = (areas > max_area) | (areas < min_area)
    return geometry.index[invalid].tolist()


class ROI(Feature):
    """A class that controls all the ROIs on the map"""

    LAYER_NAME = "ROIs"
    SELECTED_LAYER_NAME = "Selected ROIs"
    MAX_SIZE = 98000000  # 98km^2 area
    MIN_SIZE = 0

    def __init__(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        shoreline: Optional[gpd.GeoDataFrame] = None,
        rois_gdf: Optional[gpd.GeoDataFrame] = None,
        square_len_lg: float = 0,
        square_len_sm: float = 0,
        filename: Optional[str] = None,
    ):
        """
        Initializes the ROI object.

        Args:
            bbox: Bounding box GeoDataFrame.
            shoreline: Shoreline GeoDataFrame.
            rois_gdf: Existing ROIs GeoDataFrame.
            square_len_lg: Large square length for ROI generation.
            square_len_sm: Small square length for ROI generation.
            filename: Filename for saving ROIs.
        """
        # gdf : geodataframe of ROIs
        self.gdf = gpd.GeoDataFrame()
        # roi_settings : after ROIs have been downloaded holds all download settings
        self.roi_settings = {}
        # extract_shorelines : dictionary with ROIs' ids as the keys holding the extracted shorelines
        # ex. {'1': Extracted Shoreline()}
        self.extracted_shorelines = {}
        # cross_shore_distancess : dictionary with of cross-shore distance along each of the transects. Not tidally corrected.
        self.cross_shore_distances = {}
        self.filename = filename or "rois.geojson"

        if rois_gdf is not None:
            self.gdf = self._initialize_from_roi_gdf(rois_gdf)
        else:
            self.gdf = self._initialize_from_bbox_and_shoreline(
                bbox, shoreline, square_len_lg, square_len_sm
            )

    def __repr__(self):
        """
        Returns string representation of the ROI object.
        """
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf.head().to_string()
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        extracted_shoreline_info = ""
        for key in self.extracted_shorelines.keys():
            if hasattr(self.extracted_shorelines[key], "gdf") and (
                isinstance(self.extracted_shorelines[key].gdf, gpd.GeoDataFrame)
            ):
                if not self.extracted_shorelines[key].gdf.empty:
                    extracted_shoreline_info.join(
                        f"ROI ID {key}:\n{len(self.extracted_shorelines[key].gdf)}\n"
                    )
        return f"ROI:\nROI IDs: {self.get_ids()}\nROI IDs with extracted shorelines: {extracted_shoreline_info}\nROI IDs with shoreline transect intersections: {list(self.cross_shore_distances.keys())}\n gdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 5 Rows:\n{first_rows}"

    __str__ = __repr__

    def remove_by_id(
        self, ids_to_drop: Union[Collection[Union[str, int]], str, int]
    ) -> gpd.GeoDataFrame:
        """
        Removes ROIs by their IDs.

        Args:
            ids_to_drop: IDs to remove, can be single or collection.

        Returns:
            Updated GeoDataFrame after removal.
        """
        if self.gdf.empty or "id" not in self.gdf.columns or ids_to_drop is None:
            return self.gdf
        if isinstance(ids_to_drop, (str, int)):
            ids_to_drop = [
                str(ids_to_drop)
            ]  # Convert to list and ensure ids are strings
        # Ensure all elements in ids_to_drop are strings for consistent comparison
        ids_to_drop = set(map(str, ids_to_drop))
        logger.info(f"ids_to_drop from roi: {ids_to_drop}")
        # drop the ids from the geodataframe
        self.gdf = self.gdf[~self.gdf["id"].astype(str).isin(ids_to_drop)]
        # remove the corresponding extracted shorelines
        for roi_id in ids_to_drop:
            self.remove_extracted_shorelines(roi_id)

        return self.gdf

    def _initialize_from_roi_gdf(self, rois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Initializes gdf from a GeoDataFrame of ROIs.

        Args:
            rois_gdf: GeoDataFrame of ROIs.

        Returns:
            GeoDataFrame of cleaned ROIs.

        Raises:
            InvalidSize: If ROIs have invalid size.
        """
        gdf = self.clean_gdf(
            self.ensure_crs(rois_gdf),
            columns_to_keep=("id", "geometry"),
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=("Polygon", "MultiPolygon"),
            feature_type="ROI",
            unique_ids=True,
            ids_as_str=True,
        )
        # ensure all the provided ROIS are a valid size and drop any that aren't
        gdf = self.validate_ROI_sizes(gdf)
        return gdf

    def validate_ROI_sizes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        drop_index = get_ids_with_invalid_area(
            gdf, max_area=self.MAX_SIZE, min_area=self.MIN_SIZE
        )
        logger.info(f"Dropping ROIs that are an invalid size {drop_index}")
        gdf.drop(index=drop_index, axis=0, inplace=True)
        if gdf.empty:
            raise exceptions.InvalidSize(
                "The ROI(s) had an invalid size.",
                "ROI",
                max_size=ROI.MAX_SIZE,
                min_size=ROI.MIN_SIZE,
            )

        return gdf

    def _initialize_from_bbox_and_shoreline(
        self,
        bbox: Optional[gpd.GeoDataFrame],
        shoreline: Optional[gpd.GeoDataFrame],
        square_len_lg: float,
        square_len_sm: float,
    ) -> gpd.GeoDataFrame:
        """
        Initializes gdf from bounding box and shoreline.

        Args:
            bbox: Bounding box GeoDataFrame.
            shoreline: Shoreline GeoDataFrame.
            square_len_lg: Large square length.
            square_len_sm: Small square length.

        Returns:
            GeoDataFrame of ROIs with the bbox and that intersects with the shoreline.

        Raises:
            Object_Not_Found: If shoreline or bbox is None or empty.
            ValueError: If square lengths are invalid.
        """
        if (
            any(v is None for v in (bbox, shoreline))
            or (bbox is not None and bbox.empty)
            or (shoreline is not None and shoreline.empty)
        ):
            from coastseg import exceptions as ex

            raise ex.Object_Not_Found(
                "shorelines"
                if shoreline is None or (shoreline is not None and shoreline.empty)
                else "bounding box"
            )
        if square_len_sm == square_len_lg == 0:
            raise ValueError("Invalid square size for ROI. Must be greater than 0")
        return self.create_geodataframe(
            cast(gpd.GeoDataFrame, bbox),
            cast(gpd.GeoDataFrame, shoreline),
            square_len_lg,
            square_len_sm,
        )

    def get_roi_settings(
        self, roi_id: Union[str, Iterable[str]] = ""
    ) -> Dict[str, Any]:
        """
        Retrieves settings for specific ROI or all settings.

        Args:
            roi_id: ROI ID or collection of IDs. Empty string for all.

        Returns:
            Dict of settings.
        """
        if not hasattr(self, "roi_settings"):
            self.roi_settings = {}
        if roi_id is None:
            return self.roi_settings
        if isinstance(roi_id, str):
            if roi_id == "":
                logger.info(f"self.roi_settings: {self.roi_settings}")
                return self.roi_settings
            else:
                logger.info(
                    f"self.roi_settings[roi_id]: {self.roi_settings.get(roi_id, {})}"
                )
                return self.roi_settings.get(roi_id, {})
        elif isinstance(roi_id, Iterable) and not isinstance(roi_id, (str, bytes)):
            roi_settings = {}
            for id in roi_id:
                if not isinstance(id, str):
                    raise TypeError("Each ROI ID must be a string")
                if id in self.roi_settings:
                    roi_settings[id] = self.roi_settings.get(id, {})
            return roi_settings
        else:
            raise TypeError("roi_id must be a string or a collection of strings")

    def set_roi_settings(self, roi_settings: Dict[str, Any]) -> None:
        """
        Sets ROI settings dictionary.

        Args:
            roi_settings: Dict of settings.

        Raises:
            ValueError: If roi_settings is None.
            TypeError: If roi_settings is not dict.
        """
        if roi_settings is None:
            raise ValueError("roi_settings cannot be None")
        if not isinstance(roi_settings, dict):
            raise TypeError("roi_settings must be a dictionary")

        logger.info(f"Saving roi_settings {roi_settings}")
        self.roi_settings = roi_settings

    def update_roi_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates ROI settings with new values.

        Args:
            new_settings: Dict of new settings.

        Returns:
            Updated settings dict.
        """
        if new_settings is None:
            raise ValueError("new_settings cannot be None")

        logger.info(f"Updating roi_settings with {new_settings}")
        if self.roi_settings is None:
            self.roi_settings = new_settings
        else:
            self.roi_settings.update(new_settings)
        return self.roi_settings

    def get_extracted_shoreline(
        self, roi_id: str
    ) -> Optional["Extracted_Shoreline"]:  # noqa: F821
        """
        Returns extracted shoreline for ROI ID.

        Args:
            roi_id: ROI ID.

        Returns:
            Extracted shoreline or None.
        """
        return self.extracted_shorelines.get(roi_id, None)

    def get_ids(self) -> List[str]:
        """
        Returns list of all ROI IDs.
        """
        if "id" not in self.gdf.columns:
            return []
        return self.gdf["id"].tolist()

    def get_ids_with_extracted_shorelines(self) -> List[str]:
        """
        Returns list of ROI IDs with extracted shorelines.
        """
        return list(self.get_all_extracted_shorelines().keys())

    def add_geodataframe(self, gdf: gpd.GeoDataFrame) -> "ROI":
        """
        Adds GeoDataFrame to existing ROI object.

        Args:
            gdf: GeoDataFrame to add.

        Returns:
            Updated ROI object.
        """
        # check if geodataframe column has 'id' column and add one if one doesn't exist
        if "id" not in gdf.columns:
            gdf["id"] = gdf.index.astype(str).tolist()
        # drop any ROIs that are an invalid size
        gdf = self.validate_ROI_sizes(gdf)
        # Combine the two GeoDataFrames and drop duplicates from columns "id" and "geometry"
        combined_gdf = pd.concat([self.gdf, gdf], axis=0).drop_duplicates(
            subset=["id", "geometry"]
        )
        # Convert the combined DataFrame back to a GeoDataFrame
        self.gdf = gpd.GeoDataFrame(combined_gdf, crs=self.gdf.crs)
        return self

    def get_all_extracted_shorelines(
        self,
    ) -> Dict[str, "Extracted_Shoreline"]:  # noqa: F821
        """
        Returns dict of all extracted shorelines.
        """
        if not hasattr(self, "extracted_shorelines"):
            self.extracted_shorelines = {}
        return self.extracted_shorelines

    def remove_extracted_shorelines(
        self, roi_id: Optional[Union[str, int]] = None, remove_all: bool = False
    ) -> None:
        """
        Removes extracted shoreline for ROI ID or all.

        Args:
            roi_id: ROI ID to remove. None to remove specific.
            remove_all: If True, remove all shorelines.
        """
        if roi_id in self.extracted_shorelines:
            del self.extracted_shorelines[roi_id]
        if remove_all:
            del self.extracted_shorelines
            self.extracted_shorelines = {}

    def remove_selected_shorelines(
        self, roi_id: str, dates: List[datetime.datetime], satellites: List[str]
    ) -> None:
        """
        Removes selected shorelines for ROI.

        Args:
            roi_id (str): The ID of the ROI.
            dates (list[datetime.datetime]): A list of dates for which the shorelines should be removed.
            satellites (list[str]): A list of satellite names for which the shorelines should be removed.

        Returns:
            None
        """
        if roi_id in self.get_ids_with_extracted_shorelines():
            extracted_shoreline = self.get_extracted_shoreline(roi_id)
            if extracted_shoreline is not None:
                extracted_shoreline.remove_selected_shorelines(dates, satellites)

    def add_extracted_shoreline(
        self,
        extracted_shoreline: "Extracted_Shoreline",  # noqa: F821 # type: ignore
        roi_id: str,
    ) -> None:
        """
        Adds extracted shoreline for ROI ID.

        Args:
            extracted_shoreline: Extracted shoreline object.
            roi_id: ROI ID.
        """
        self.extracted_shorelines[roi_id] = extracted_shoreline
        logger.info(f"New extracted shoreline added for ROI {roi_id}")
        # logger.info(f"New extracted shoreline added for ROI {roi_id}: {self.extracted_shorelines}")

    def get_cross_shore_distances(self, roi_id: str) -> Dict[str, Any]:
        """
        Returns cross shore distances for ROI ID.

        Args:
            roi_id: ROI ID.

        Returns:
            Dict of cross shore distances.
        """
        result = self.cross_shore_distances.get(roi_id, {})
        if result == {}:
            logger.info(f"ROI: {roi_id} has no cross shore distance")
        else:
            logger.info(f"ROI: {roi_id} cross distance with keys : {result}")
        return result

    def add_cross_shore_distances(
        self, cross_shore_distance: Dict[str, Any], roi_id: str
    ) -> None:
        """
        Adds cross shore distances for ROI ID.

        Args:
            cross_shore_distance: Dict of distances.
            roi_id: ROI ID.
        """
        self.cross_shore_distances[roi_id] = cross_shore_distance

    def get_all_cross_shore_distances(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns dict of all cross shore distances.
        """
        return self.cross_shore_distances

    def remove_cross_shore_distance(
        self, roi_id: Optional[str] = None, remove_all: bool = False
    ) -> None:
        """
        Removes cross shore distance for ROI ID or all.

        Args:
            roi_id: ROI ID to remove.
            remove_all: If True, remove all distances.
        """
        if roi_id in self.cross_shore_distances:
            del self.cross_shore_distances[roi_id]
        if remove_all:
            self.cross_shore_distances = {}

    def create_geodataframe(
        self,
        bbox: gpd.GeoDataFrame,
        shoreline: gpd.GeoDataFrame,
        large_length: float = 7500,
        small_length: float = 5000,
    ) -> gpd.GeoDataFrame:
        """
        Generates ROIs along shoreline using fishnet method.

        Args:
            bbox: Bounding box GeoDataFrame.
            shoreline: Shoreline GeoDataFrame.
            large_length: Large square length.
            small_length: Small square length.

        Returns:
            GeoDataFrame of ROIs.
        """
        if bbox is None or shoreline is None:
            raise ValueError("bbox and shoreline must not be None")
        # Create a single set of fishnets with square size = small and/or large side lengths that overlap each other
        # logger.info(f"Small Length: {small_length}  Large Length: {large_length}")
        if small_length == 0 or large_length == 0:
            # create a fishnet geodataframe with square size of either large_length or small_length
            fishnet_size = large_length if large_length != 0 else small_length
            fishnet_intersect_gdf = self.get_fishnet_gdf(bbox, shoreline, fishnet_size)
        else:
            # Create two fishnets, one big (2000m) and one small(1500m) so they overlap each other
            fishnet_gpd_large = self.get_fishnet_gdf(bbox, shoreline, large_length)
            fishnet_gpd_small = self.get_fishnet_gdf(bbox, shoreline, small_length)
            # Concat the fishnets together to create one overlapping set of rois
            fishnet_intersect_gdf = gpd.GeoDataFrame(
                pd.concat([fishnet_gpd_large, fishnet_gpd_small], ignore_index=True)
            )
        # clean the geodataframe and create unique ids for each roi
        gdf = self.clean_gdf(
            fishnet_intersect_gdf,
            columns_to_keep=[
                "id",
                "geometry",
            ],
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=("Polygon", "MultiPolygon"),
            feature_type="rois",
            unique_ids=True,
            ids_as_str=True,
        )

        return gdf

    def style_layer(self, geojson: Dict[str, Any], layer_name: str) -> GeoJSON:
        """
        Returns styled GeoJSON layer.

        Args:
            geojson: GeoJSON dict.
            layer_name: Layer name.

        Returns:
            Styled GeoJSON layer.
        """
        return super().style_layer(
            geojson,
            layer_name,
            hover_style={"color": "red", "fillOpacity": 0.1, "fillColor": "crimson"},
        )

    def fishnet_intersection(
        self, fishnet: gpd.GeoDataFrame, data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Returns intersection of fishnet and data.

        Args:
            fishnet: Fishnet GeoDataFrame.
            data: Data GeoDataFrame.

        Returns:
            Intersected GeoDataFrame.
        """
        # Perform a spatial join between the fishnet and data to find the intersecting geometries
        intersection_gdf = gpd.sjoin(
            left_df=fishnet, right_df=data, how="inner", predicate="intersects"
        )

        # Remove unnecessary columns, keeping only the geometry column
        columns_to_keep = ["geometry"]
        intersection_gdf = intersection_gdf[columns_to_keep]

        # Remove duplicate geometries
        intersection_gdf.drop_duplicates(
            keep="first", subset=["geometry"], inplace=True
        )

        return intersection_gdf

    def create_rois(
        self,
        bbox: gpd.GeoDataFrame,
        square_size: float,
        input_espg: Union[str, int] = "epsg:4326",
        output_epsg: str = "epsg:4326",
    ) -> gpd.GeoDataFrame:
        """
        Creates fishnet of square ROIs.

        Args:
            bbox: Bounding box GeoDataFrame.
            square_size: Side length in meters.
            input_espg: Input EPSG.
            output_epsg: Output EPSG.

        Returns:
            GeoDataFrame of ROIs.
        """
        projected_espg = common.get_epsg_from_geometry(bbox.iloc[0]["geometry"])
        logger.info(f"ROI: projected_espg_code: {projected_espg}")
        # project geodataframe to new CRS specified by utm_code
        projected_bbox_gdf = bbox.to_crs(projected_espg)
        # create fishnet of rois
        fishnet = self.create_fishnet(
            projected_bbox_gdf, projected_espg, output_epsg, square_size
        )
        return fishnet

    def create_fishnet(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        input_espg: Union[str, int],
        output_epsg: str,
        square_size: float = 1000,
    ) -> gpd.GeoDataFrame:
        """
        Returns fishnet of ROIs intersecting bbox.

        Args:
            bbox_gdf: Bounding box GeoDataFrame.
            input_espg: Input EPSG.
            output_epsg: Output EPSG.
            square_size: Square side length in meters.

        Returns:
            Fishnet GeoDataFrame.
        """
        minX, minY, maxX, maxY = bbox_gdf.total_bounds
        # Create a fishnet where each square has side length = square size
        x, y = (minX, minY)
        geom_array = []
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon(
                    [
                        (x, y),
                        (x, y + square_size),
                        (x + square_size, y + square_size),
                        (x + square_size, y),
                        (x, y),
                    ]
                )
                # add each square to geom_array
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size

        # create geodataframe to hold all the (rois)squares
        fishnet = gpd.GeoDataFrame(geom_array, columns=["geometry"]).set_crs(input_espg)
        logger.info(
            f"\n ROIs area before conversion to {output_epsg}:\n {fishnet.area} for CRS: {input_espg}"
        )
        fishnet = fishnet.to_crs(output_epsg)
        return fishnet

    def get_fishnet_gdf(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        shoreline_gdf: gpd.GeoDataFrame,
        square_length: float = 1000,
    ) -> gpd.GeoDataFrame:
        """
        Returns fishnet intersecting shoreline.

        Args:
            bbox_gdf: Bounding box GeoDataFrame.
            shoreline_gdf: Shoreline GeoDataFrame.
            square_length: Square side length.

        Returns:
            Intersected fishnet GeoDataFrame.
        """
        if bbox_gdf is None or shoreline_gdf is None:
            raise ValueError("bbox_gdf and shoreline_gdf must not be None")
        # Get the geodataframe for the fishnet within the bbox
        fishnet = self.create_rois(bbox_gdf, square_length)
        # Get the geodataframe for the fishnet intersecting the shoreline
        fishnet_intersection = self.fishnet_intersection(fishnet, shoreline_gdf)
        return fishnet_intersection
