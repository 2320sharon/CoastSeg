"""Bounding box feature utilities for CoastSeg.

Defines the Bounding_Box class for managing rectangular areas of interest (AOI).
Supports initialization from GeoJSON or GeoDataFrame, validates geometry types,
normalizes to EPSG:4326, and provides styled map layers with area constraints.

Example:
    >>> from coastseg.bbox import Bounding_Box
    >>> import geopandas as gpd
    >>> # Create from GeoJSON polygon
    >>> polygon = {
    ...     "type": "Polygon",
    ...     "coordinates": [[[-118.5, 33.5], [-118.5, 33.6], [-118.4, 33.6], [-118.4, 33.5], [-118.5, 33.5]]]
    ... }
    >>> bbox = Bounding_Box(rectangle=polygon)
    >>> print(f"Area: {bbox.gdf.to_crs('EPSG:3857').area.iloc[0]:.2f} m²")
    >>> map_layer = bbox.style_layer(bbox.gdf.__geo_interface__, "My AOI")
"""

# Standard library imports
from typing import Any, Dict, Optional, Union

# External dependencies imports
import geopandas as gpd
from ipyleaflet import GeoJSON
from shapely.geometry import shape

from coastseg.common import preprocess_geodataframe, validate_geometry_types
from coastseg.feature import Feature

# Internal dependencies imports
from .exceptions import BboxTooLargeError, BboxTooSmallError

__all__ = ["Bounding_Box"]


class Bounding_Box(Feature):
    """A user-drawn rectangular Area of Interest (AOI).

    Wraps a GeoDataFrame that stores a single Polygon/MultiPolygon representing a
    bounding box, normalized to EPSG:4326 and validated for geometry type.

    Attributes:
        gdf (gpd.GeoDataFrame): The underlying GeoDataFrame (single geometry) with
            CRS set to EPSG:4326.
        filename (str): Default filename to use when persisting the feature.
        MAX_AREA (int): Maximum allowed area of the box in square meters.
        MIN_AREA (int): Minimum allowed area of the box in square meters.
        LAYER_NAME (str): Default display name for map layers.

    Raises:
        Exception: If rectangle is neither a GeoDataFrame nor a GeoJSON-like
            mapping.
    """

    MAX_AREA: int = 100000000000  # UNITS = Sq. Meters
    MIN_AREA: int = 1000  # UNITS = Sq. Meters
    LAYER_NAME: str = "Bbox"

    def __init__(
        self,
        rectangle: Union[Dict[str, Any], gpd.GeoDataFrame],
        filename: str = "bbox.geojson",
    ) -> None:
        """
        Initialize Bounding_Box from geometry.

        Args:
            rectangle (Dict[str, Any] | gpd.GeoDataFrame): The bounding geometry. When a dict
                is provided, it must be a GeoJSON-like mapping representing a single
                Polygon or MultiPolygon in EPSG:4326 (or accompanied by the
                provided crs in :meth:`create_geodataframe`). When a GeoDataFrame
                is provided, it will be cleaned and reprojected to EPSG:4326.
            filename (str): Default filename.

        Raises:
            Exception: If rectangle type invalid.
        """
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.filename: str = filename
        if isinstance(rectangle, gpd.GeoDataFrame):
            self.gdf = self._initialize_from_gdf(rectangle)
        elif isinstance(rectangle, dict):
            self.gdf = self.create_geodataframe(rectangle)
        else:
            raise Exception(
                "Invalid rectangle provided to BBox must be either a geodataframe or dict"
            )

    def __str__(self) -> str:
        """Return string representation."""
        return f"BBox: geodataframe {self.gdf}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BBox: geodataframe {self.gdf}"

    def _initialize_from_gdf(self, bbox_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clean and validate GeoDataFrame as bounding box.

        Args:
            bbox_gdf: Input GeoDataFrame.

        Returns:
            Cleaned GeoDataFrame in EPSG:4326.
        """
        # clean the geodataframe
        bbox_gdf = preprocess_geodataframe(
            bbox_gdf,
            columns_to_keep=["geometry"],
            create_ids=False,
            output_crs="EPSG:4326",
        )
        validate_geometry_types(
            bbox_gdf, set(["Polygon", "MultiPolygon"]), feature_type="Bounding Box"
        )
        return bbox_gdf

    def create_geodataframe(
        self, rectangle: Dict[str, Any], crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Create GeoDataFrame from GeoJSON mapping.

        Args:
            rectangle: GeoJSON dict of geometry.
            crs: Input CRS. Defaults to "EPSG:4326".

        Returns:
            GeoDataFrame in EPSG:4326.
        """
        geom = [shape(rectangle)]
        geojson_bbox = gpd.GeoDataFrame({"geometry": geom})
        geojson_bbox.set_crs(crs, inplace=True)
        # clean the geodataframe
        geojson_bbox = preprocess_geodataframe(
            geojson_bbox,
            columns_to_keep=["geometry"],
            create_ids=False,
            output_crs="EPSG:4326",
        )
        validate_geometry_types(
            geojson_bbox, set(["Polygon", "MultiPolygon"]), feature_type="Bounding Box"
        )
        return geojson_bbox

    def style_layer(self, geojson: Dict[str, Any], layer_name: str) -> GeoJSON:
        """
        Return styled GeoJSON layer for map.

        Args:
            geojson: GeoJSON to render.
            layer_name: Display name.

        Returns:
            Styled GeoJSON layer.
        """
        style = {
            "color": "#75b671",
            "fill_color": "#75b671",
            "opacity": 1,
            "fillOpacity": 0.1,
            "weight": 3,
        }
        return super().style_layer(geojson, layer_name, style=style, hover_style={})

    @staticmethod
    def check_bbox_size(bbox_area: Union[int, float]) -> None:
        """
        Validate bounding box area.

        Args:
            bbox_area: Area in square meters.

        Raises:
            BboxTooLargeError: If exceeds max area.
            BboxTooSmallError: If below min area.
        """
        if bbox_area > Bounding_Box.MAX_AREA:
            raise BboxTooLargeError()
        elif bbox_area < Bounding_Box.MIN_AREA:
            raise BboxTooSmallError()
