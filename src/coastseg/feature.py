from __future__ import annotations

import json
from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import geopandas as gpd
import pandas as pd
from ipyleaflet import GeoJSON

# project helpers
from coastseg.common import (
    create_unique_ids,
    preprocess_geodataframe,
    validate_geometry_types,
)


class Feature(ABC):
    """Lightweight base for geospatial features with common utilities."""

    DEFAULT_CRS = "EPSG:4326"
    FILE_EXT = ".geojson"

    # ------- core state -------
    def __init__(self, filename: Optional[str] = None) -> None:
        self.gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.filename = filename or f"{self.__class__.__name__.lower()}{self.FILE_EXT}"

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Filename must be a string.")
        if not value.endswith(self.FILE_EXT):
            raise ValueError(f"Filename must end with '{self.FILE_EXT}'.")
        self._filename = value

    # ------- minimal, readable repr/str shared by children -------
    def __repr__(self) -> str:  # one method covers both; readable + stable
        crs = getattr(self.gdf, "crs", None)
        head = self.gdf.head().to_string() if not self.gdf.empty else "<empty>"
        dtypes = (
            self.gdf.dtypes.apply(lambda x: x.name).to_dict()
            if not self.gdf.empty
            else {}
        )
        return (
            f"{self.__class__.__name__}("
            f"CRS={crs}, "
            f"columns={list(self.gdf.columns)}, "
            f"dtypes={dtypes}, "
            f"preview=\n{head}\n)"
        )

    __str__ = __repr__

    # ------- input normalization -------
    @staticmethod
    def gdf_from_mapping(
        mapping: Dict[str, Any], crs: str = DEFAULT_CRS
    ) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame from a GeoJSON-like mapping dictionary."""
        from shapely.geometry import shape

        gdf = gpd.GeoDataFrame({"geometry": [shape(mapping)]}, crs=crs)
        return gdf

    @staticmethod
    def drop_all_na_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return gdf.dropna(axis=1, how="all")

    @staticmethod
    def ensure_crs(gdf: gpd.GeoDataFrame, crs: str = DEFAULT_CRS) -> gpd.GeoDataFrame:
        return gdf.to_crs(crs) if getattr(gdf, "crs", None) else gdf.set_crs(crs)

    # ------- style & geojson helpers -------
    @staticmethod
    def to_geojson(data: Union[Dict[str, Any], gpd.GeoDataFrame]) -> Dict[str, Any]:
        if isinstance(data, dict):
            return data
        return json.loads(data.to_json())

    # ---- ids + removal are common across classes ----
    def ids(self) -> list[str]:
        return (
            []
            if self.gdf.empty or "id" not in self.gdf.columns
            else self.gdf["id"].astype(str).tolist()
        )

    def remove_by_id(
        self, ids_to_drop: Union[List[str], set, tuple, str, int]
    ) -> gpd.GeoDataFrame:
        """
        Remove features by their IDs.

        Args:
            ids_to_drop (Union[List[str], Set[str], Tuple[str, ...], str, int]):
                IDs of features to remove. Can be a single ID or collection of IDs.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with specified features removed.

        Note:
            Returns the original GeoDataFrame unchanged if it's empty, has no 'id' column,
            or ids_to_drop is None.
        """
        if self.gdf.empty or "id" not in self.gdf.columns or ids_to_drop is None:
            return self.gdf
        if isinstance(ids_to_drop, (str, int)):
            ids_to_drop = [str(ids_to_drop)]
        ids_to_drop = set(map(str, ids_to_drop))
        # drop the ids from the geodataframe
        self.gdf = self.gdf[~self.gdf["id"].astype(str).isin(ids_to_drop)]
        return self.gdf

    @staticmethod
    def concat_clean(
        gdfs: List[gpd.GeoDataFrame],
        ignore_index: bool = True,
        drop_all_na: bool = True,
    ) -> gpd.GeoDataFrame:
        if drop_all_na:
            gdfs = [df.dropna(axis=1, how="all") for df in gdfs]
        return (
            pd.concat(gdfs, ignore_index=ignore_index) if gdfs else gpd.GeoDataFrame()
        )

    # read→preprocess→validate→(optional) clip; perfect for Shoreline.get_clipped_shoreline
    @classmethod
    def read_masked_clean(
        cls,
        path: str,
        *,
        mask: Optional[gpd.GeoDataFrame] = None,
        columns_to_keep: Iterable[str] = ("geometry",),
        geometry_types: Optional[Iterable[str]] = None,
        feature_type: str = "feature",
        output_crs: str = DEFAULT_CRS,
    ) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path, mask=mask)
        gdf = cls.clean_gdf(
            gdf,
            columns_to_keep=columns_to_keep,
            output_crs=output_crs,
            geometry_types=geometry_types,
            feature_type=feature_type,
        )
        return gpd.clip(gdf, mask) if mask is not None else gdf

    # ------- single entry point for cleaning/validating -------
    @staticmethod
    def clean_gdf(
        gdf: gpd.GeoDataFrame,
        *,
        columns_to_keep: Iterable[str] = ("geometry",),
        output_crs: str = DEFAULT_CRS,
        create_ids_flag: bool = False,
        geometry_types: Optional[Iterable[str]] = None,
        feature_type: str = "feature",
        unique_ids: bool = False,
        ids_as_str: bool = False,
        help_message: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        gdf = preprocess_geodataframe(
            gdf,
            columns_to_keep=list(columns_to_keep),
            create_ids=create_ids_flag,
            output_crs=output_crs,
        )
        if geometry_types:
            validate_geometry_types(
                gdf,
                set(geometry_types),
                feature_type=feature_type,
                help_message=help_message,
            )
        if unique_ids:
            gdf = create_unique_ids(gdf, prefix_length=3)
        if ids_as_str and "id" in gdf.columns:
            gdf["id"] = gdf["id"].astype(str)
        return gdf

    def style_layer(
        self,
        data: Union[Dict[str, Any], gpd.GeoDataFrame],
        layer_name: str,
        *,
        style: Optional[Dict[str, Any]] = None,
        hover_style: Optional[Dict[str, Any]] = None,
    ) -> GeoJSON:
        if style is None:
            style = {
                "color": "#555555",
                "fill_color": "#555555",
                "fillOpacity": 0.1,
                "weight": 1,
            }
        if hover_style is None:
            hover_style = {}
        geojson = self.to_geojson(data)
        if not geojson:
            raise ValueError(f"Empty {layer_name} geojson cannot be drawn.")
        return GeoJSON(
            data=geojson, name=layer_name, style=style, hover_style=hover_style
        )

    # ------- generic area validator (children supply limits/exceptions) -------
    @staticmethod
    def check_size(
        area_m2: Union[int, float],
        *,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        too_small_exc: Optional[Type[Exception]] = None,
        too_large_exc: Optional[Type[Exception]] = None,
    ) -> None:
        if max_area is not None and area_m2 > max_area and too_large_exc:
            raise too_large_exc()
        if min_area is not None and area_m2 < min_area and too_small_exc:
            raise too_small_exc()
