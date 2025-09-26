"""
CoastSeg Shoreline Management.

Tools to download, validate, clip, and visualize shoreline geometries
(LineString/MultiLineString) — including CRS handling (EPSG:4326), unique ID
generation, and simple ipyleaflet styling. Supports fetching data from Zenodo.

Classes:
  - Shoreline: manages shoreline data and operations.

Functions:
  - construct_download_url
  - get_intersecting_files
  - load_total_bounds_df
"""

# Standard library imports
import logging
import os
from typing import Any, Callable, Dict, List, Optional

# External dependencies imports
import geopandas as gpd
import pandas as pd
from ipyleaflet import GeoJSON

# Internal dependencies imports
from coastseg import exception_handler
from coastseg.common import (
    download_url,
)
from coastseg.exceptions import DownloadError
from coastseg.feature import Feature

logger = logging.getLogger(__name__)

# only export Shoreline class
__all__ = ["Shoreline"]


class Shoreline(Feature):
    """
    Represents shoreline data within a specified region.

    This class manages shoreline geometries (LineStrings and MultiLineStrings) within
    a bounding box, providing functionality for downloading, preprocessing, clipping,
    and styling shoreline data from various sources including Zenodo datasets.

    Attributes:
        LAYER_NAME (str): Default layer name for shoreline display.
        SELECTED_LAYER_NAME (str): Layer name for selected shorelines.
        gdf (gpd.GeoDataFrame): GeoDataFrame containing shoreline geometries.
        filename (str): Name of the shoreline file.
    """

    LAYER_NAME = "shoreline"
    SELECTED_LAYER_NAME = "Selected Shorelines"
    # Read in each shoreline file and clip it to the bounding box
    COLUMNS_TO_KEEP = [
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

    def __init__(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        shoreline: Optional[gpd.GeoDataFrame] = None,
        filename: Optional[str] = None,
        download_location: Optional[str] = None,
    ) -> None:
        """
        Initialize a Shoreline object with optional data sources.

        Args:
            bbox (Optional[gpd.GeoDataFrame]): Bounding box to clip shorelines to.
                If provided, shorelines will be downloaded and clipped to this area.
            shoreline (Optional[gpd.GeoDataFrame]): Existing shoreline data to use.
                Takes precedence over bbox if both are provided.
            filename (Optional[str]): Name for the shoreline file. Must end with '.geojson'.
                Defaults to 'shoreline.geojson'.

            download_location (Optional[str]): Directory for downloading shoreline files.
                Defaults to the module directory.

        Raises:
            ValueError: If shoreline data is invalid or bbox processing fails.
        """
        # location to create shorelines directory and download shorelines to
        self._download_location = download_location or os.path.dirname(
            os.path.abspath(__file__)
        )
        # initialize the shorelines
        super().__init__(filename or "shoreline.geojson")
        self.initialize_shorelines(bbox, shoreline)

    def __repr__(self) -> str:
        """
        Return a concise string representation of the Shoreline object.
        Provides detailed information about the shoreline data including CRS,
        column information, first few rows, geometry preview, and IDs.

        Returns:
            str: Same as __str__ method for detailed representation.
        """
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 3 rows as a string
        first_rows = self.gdf
        geom_str = ""
        if isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                first_rows = self.gdf.head(3).drop(columns="geometry").to_string()
            if not self.gdf.empty:
                geom_str = str(self.gdf.iloc[0]["geometry"])[:100] + "...)"
        # Get CRS information
        if self.gdf.empty:
            crs_info = "CRS: None"
        else:
            if self.gdf is not None and hasattr(self.gdf, "crs"):
                crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
            else:
                crs_info = "CRS: None"

        ids = []
        if "id" in self.gdf.columns:
            ids = self.gdf["id"].astype(str)
        return f"Shoreline:\nself.gdf:\n\n{crs_info}\n- Columns and Data Types:\n{col_info}\n\n- First 3 Rows:\n{first_rows}\n geometry: {geom_str}\nIDs:\n{ids}"

    __str__ = __repr__

    def initialize_shorelines(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        shorelines: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """
        Initialize shoreline data from either existing shorelines or a bounding box.

        Args:
            bbox (Optional[gpd.GeoDataFrame]): Bounding box to download and clip shorelines.
            shorelines (Optional[gpd.GeoDataFrame]): Existing shoreline data to use directly.

        Note:
            If both bbox and shorelines are provided, shorelines takes precedence.
        """
        if shorelines is not None:
            self.gdf = self.initialize_shorelines_with_shorelines(shorelines)

        elif bbox is not None:
            self.gdf = self.initialize_shorelines_with_bbox(bbox)

    def initialize_shorelines_with_shorelines(
        self, shorelines: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Initialize shorelines using provided shoreline GeoDataFrame.

        Validates and preprocesses the provided shorelines, ensuring they have the correct
        geometry types (LineString or MultiLineString) and proper CRS (EPSG:4326).

        Args:
            shorelines (gpd.GeoDataFrame): GeoDataFrame containing shoreline geometries.

        Raises:
            ValueError: If shorelines is not a GeoDataFrame.

        Note:
            - Sets CRS to EPSG:4326 if not already set
            - Validates geometry types are LineString or MultiLineString
            - Creates unique IDs for all features
            - Keeps only specified columns related to shoreline properties
        """
        if not isinstance(shorelines, gpd.GeoDataFrame):
            raise ValueError("Shorelines must be a GeoDataFrame.")
        if shorelines.empty:
            logger.warning("Shorelines cannot be empty.")
            return shorelines

        return self.clean_gdf(
            self.ensure_crs(shorelines, self.DEFAULT_CRS),
            columns_to_keep=self.COLUMNS_TO_KEEP,
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=["LineString", "MultiLineString"],
            feature_type="shoreline",
            unique_ids=True,
            ids_as_str=True,
            help_message="The uploaded shorelines need to be LineStrings.",
        )

    def initialize_shorelines_with_bbox(
        self, bbox: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Initialize shorelines by downloading and clipping to a bounding box.

        Downloads shoreline data that intersects with the provided bounding box and
        clips the shorelines to the bbox boundaries.

        Args:
            bbox (gpd.GeoDataFrame): Bounding box geometry for clipping shorelines.

        Raises:
            FileNotFoundError: If no shoreline files are found that intersect the bbox.
            Exception: If no default shoreline features are available.
        """
        if bbox is None or bbox.empty:
            raise ValueError(
                "Cannot create shoreline within a Bounding box thats empty or None."
            )
        shoreline_files = self.get_intersecting_shoreline_files(bbox)
        # if no shorelines were found to intersect with the bounding box raise an exception
        if not shoreline_files:
            exception_handler.check_if_default_feature_available(None, "shoreline")

        return self.create_geodataframe(bbox, shoreline_files)

    def get_clipped_shoreline(
        self, shoreline_file: str, bbox: gpd.GeoDataFrame, columns_to_keep: List[str]
    ) -> gpd.GeoDataFrame:
        """
        Read, preprocess, and clip a shoreline file to the bounding box.

        Args:
            shoreline_file (str): Path to the shoreline file to read.
            bbox (gpd.GeoDataFrame): Bounding box for clipping.
            columns_to_keep (List[str]): List of column names to retain.

        Returns:
            gpd.GeoDataFrame: Clipped shoreline GeoDataFrame with validated geometries.

        Note:
            Validates that geometries are LineString or MultiLineString types.
        """
        return Feature.read_masked_clean(
            shoreline_file,
            mask=bbox,
            columns_to_keep=columns_to_keep,
            geometry_types=["LineString", "MultiLineString"],
            feature_type="shoreline",
            output_crs=self.DEFAULT_CRS,
        )

    def get_intersecting_shoreline_files(
        self, bbox: gpd.GeoDataFrame, bounding_boxes_location: str = ""
    ) -> List[str]:
        """
        Retrieves a list of intersecting shoreline files based on the given bounding box.

        Args:
            bbox (gpd.GeoDataFrame): The bounding box to use for finding intersecting shoreline files.
            bounding_boxes_location (str, optional): The location to store the bounding box files. If not provided,
                it defaults to the download location specified during object initialization.

        Returns:
            List[str]: A list of intersecting shoreline file paths.

        Raises:
            ValueError: If no intersecting shorelines were available within the bounding box.
            FileNotFoundError: If no shoreline files were found at the download location.
        """
        # load the intersecting shoreline files
        bounding_boxes_location = (
            bounding_boxes_location
            if bounding_boxes_location
            else os.path.join(self._download_location, "bounding_boxes")
        )
        os.makedirs(bounding_boxes_location, exist_ok=True)
        intersecting_files = get_intersecting_files(bbox, bounding_boxes_location)

        if not intersecting_files:
            logger.warning("No intersecting shoreline files were found.")
            return []

        # Download any missing shoreline files
        shoreline_files = self.get_shoreline_files(
            intersecting_files, self._download_location
        )
        if not shoreline_files:
            raise FileNotFoundError(
                f"No shoreline files were found at {self._download_location}."
            )
        return shoreline_files

    def create_geodataframe(
        self, bbox: gpd.GeoDataFrame, shoreline_files: List[str], crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame containing shorelines that intersect with the bounding box.

        Reads multiple shoreline files, clips them to the bounding box, and combines
        them into a single GeoDataFrame with unique IDs and proper CRS.

        Args:
            bbox (gpd.GeoDataFrame): Bounding box for clipping shorelines.
            shoreline_files (List[str]): List of file paths to shoreline data.
            crs (str): Target coordinate reference system. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: Combined shoreline data clipped to bbox with unique IDs.

        Raises:
            FileNotFoundError: If no shoreline files are provided.

        Note:
            - Removes columns where all values are NA before concatenation
            - Validates all geometries are LineString or MultiLineString
            - Creates unique IDs with 3-character prefixes
        """
        # Read in each shoreline file and clip it to the bounding box
        if not shoreline_files:
            logger.error("No shoreline files were provided to read shorelines from")
            raise FileNotFoundError(
                f"No shoreline files were found at {self._download_location}."
            )

        shorelines_gdf = gpd.GeoDataFrame()
        shorelines = [
            self.get_clipped_shoreline(file, bbox, self.COLUMNS_TO_KEEP)
            for file in shoreline_files
        ]
        # Drop columns where all values are NA
        shorelines = [df.dropna(axis=1, how="all") for df in shorelines]

        # Concatenate the DataFrames
        shorelines_gdf = pd.concat(shorelines, ignore_index=True)

        return self.clean_gdf(
            shorelines_gdf,
            columns_to_keep=list(self.COLUMNS_TO_KEEP),
            output_crs=crs,
            create_ids_flag=True,
            geometry_types=("LineString", "MultiLineString"),
            feature_type="transects",
            unique_ids=True,
            ids_as_str=True,
            help_message="The uploaded transects need to be LineStrings.",
        )

    def get_shoreline_files(
        self, intersecting_shoreline_files: Dict[str, str], download_location: str
    ) -> List[str]:
        """
        Download missing shoreline files and return list of available file paths.

        Checks for existing shoreline files and downloads any missing ones from
        the specified dataset sources (e.g., Zenodo).

        Args:
            intersecting_shoreline_files (Dict[str, str]): Dictionary mapping
                shoreline filenames to their dataset IDs.
            download_location (str): Full path to location where the shorelines
                directory should be created.

        Returns:
            List[str]: List of file paths for all available shoreline files.

        Note:
            - Creates a 'shorelines' directory if it doesn't exist
            - Skips download if file already exists locally
            - Logs download errors but continues processing other files
        """
        available_files = []

        # Ensure the directory to hold the downloaded shorelines from Zenodo exists
        shoreline_dir = os.path.abspath(os.path.join(download_location, "shorelines"))
        os.makedirs(shoreline_dir, exist_ok=True)

        for filename, dataset_id in intersecting_shoreline_files.items():
            shoreline_path = os.path.join(shoreline_dir, filename)
            if not os.path.exists(shoreline_path):
                try:
                    self.download_shoreline(filename, shoreline_path, dataset_id)
                    available_files.append(shoreline_path)
                except DownloadError as download_exception:
                    logger.error(
                        f"{download_exception} Shoreline {filename} failed to download."
                    )
                    print(
                        f"{download_exception} Shoreline {filename} failed to download."
                    )
            else:
                # if the shoreline file already exists then add it to the list of available files
                available_files.append(shoreline_path)
        return available_files

    def style_layer(self, geojson: Dict[str, Any], layer_name: str) -> GeoJSON:
        """
        Return a styled GeoJSON layer for shoreline visualization.

        Creates a black dashed line style appropriate for shoreline display
        on interactive maps with hover effects.

        Args:
            geojson (Dict[str, Any]): GeoJSON dictionary containing shoreline data.
            layer_name (str): Name for the GeoJSON layer.

        Returns:
            Any: Styled GeoJSON layer (ipyleaflet.GeoJSON) ready for map display.
        """
        style = {
            "color": "black",
            "fill_color": "black",
            "opacity": 1,
            "dashArray": "5",
            "fillOpacity": 0.5,
            "weight": 4,
        }
        hover_style = {"color": "white", "dashArray": "4", "fillOpacity": 0.7}
        return super().style_layer(
            geojson, layer_name, style=style, hover_style=hover_style
        )

    def download_shoreline(
        self,
        filename: str,
        save_location: str,
        dataset_id: str = "7814755",
        download_function: Callable = download_url,
    ) -> None:
        """
        Download shoreline files from Zenodo dataset.

        Constructs the download URL and downloads the specified shoreline file
        from the Zenodo repository to the local filesystem.

        Args:
            filename (str): Name of the file to download from Zenodo.
            save_location (str): Full path where the downloaded file should be saved.
            dataset_id (str): Zenodo dataset ID. Defaults to '7814755' (world shorelines).

        Note:
            - Uses the configured download service for actual file transfer
            - Logs the download operation for monitoring
            - Raises DownloadError if the download fails
        """

        # Construct the download URL
        root_url = "https://zenodo.org/record/"
        url = construct_download_url(root_url, dataset_id, filename)

        # Download shorelines from Zenodo
        logger.info(f"Retrieving file: {save_location} from {url}")
        download_function(url, save_location, filename)


# helper functions
def construct_download_url(root_url: str, dataset_id: str, filename: str) -> str:
    """
    Construct a download URL for Zenodo dataset files.

    Args:
        root_url (str): Base URL for the Zenodo repository (e.g., "https://zenodo.org/record/").
        dataset_id (str): Zenodo dataset identifier.
        filename (str): Name of the file to download.

    Returns:
        str: Complete download URL with download parameter.

    Example:
        >>> construct_download_url("https://zenodo.org/record/", "7814755", "shoreline.geojson")
        "https://zenodo.org/record/7814755/files/shoreline.geojson?download=1"
    """
    return f"{root_url}{dataset_id}/files/{filename}?download=1"


def get_intersecting_files(
    bbox: gpd.GeoDataFrame, bounding_boxes_location: str
) -> Dict[str, str]:
    """
    Find shoreline files that intersect with the given bounding box.

    Searches through available shoreline bounding box datasets to identify
    which shoreline files contain data that intersects with the provided bbox.

    Args:
        bbox (gpd.GeoDataFrame): Bounding box geometry to search for intersections.
        bounding_boxes_location (str): Full path to the directory containing
            bounding box datasets.

    Returns:
        Dict[str, str]: Dictionary mapping intersecting shoreline filenames
            to their dataset IDs.

    Note:
        - Currently searches world shoreline dataset (ID: 7814755)
        - Uses spatial intersection to determine file relevance
        - Returns empty dict if no intersecting files are found
    """
    WORLD_DATASET_ID = "7814755"

    # DataFrames containing total bounding box for each shoreline file
    world_total_bounds_df = load_total_bounds_df(bounding_boxes_location, "world", bbox)
    # Create a list of tuples containing the DataFrames and their dataset IDs
    total_bounds_dfs = [
        (world_total_bounds_df, WORLD_DATASET_ID),
    ]

    intersecting_files = {}
    # Add filenames of interesting shoreline in both the usa and world shorelines to intersecting_files
    for bounds_df, dataset_id in total_bounds_dfs:
        if not bounds_df.empty:
            filenames = bounds_df.index
            # Create a dictionary mapping filenames to their dataset IDs
            filenames_and_ids = zip(filenames, [dataset_id] * len(filenames))
            # Add the filenames and their dataset IDs to intersecting_files
            intersecting_files.update(dict(filenames_and_ids))
    logger.debug(
        f"Found {len(intersecting_files)} intersecting files\n {intersecting_files}"
    )
    return intersecting_files


def load_total_bounds_df(
    bounding_boxes_location: str,
    location: str = "usa",
    mask: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    """
    Load and return bounding boxes for shoreline datasets.

    Reads GeoJSON files containing bounding box information for shoreline
    datasets and optionally filters them using a spatial mask.

    Args:
        bounding_boxes_location (str): Full path to the directory containing
            bounding box files.
        location (str): Dataset location identifier. Either 'world' or 'usa'.
            Defaults to 'usa'.
        mask (Optional[gpd.GeoDataFrame]): Optional spatial mask for filtering
            bounding boxes. Only boxes intersecting this mask will be returned.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with shoreline bounding boxes indexed by filename.

    Note:
        - For 'usa': loads 'usa_shorelines_bounding_boxes.geojson'
        - For 'world': loads 'world_reference_shorelines_bboxes.geojson'
        - Sets index to filename column for easy lookup
        - Removes filename column after setting as index
    """
    # load in different  total bounding box different depending on location given
    if location == "usa":
        gdf_file = "usa_shorelines_bounding_boxes.geojson"
    elif location == "world":
        gdf_file = "world_reference_shorelines_bboxes.geojson"

    gdf_location = os.path.join(bounding_boxes_location, gdf_file)
    total_bounds_df = gpd.read_file(gdf_location, mask=mask)
    total_bounds_df.index = total_bounds_df["filename"]  # type: ignore
    if "filename" in total_bounds_df.columns:
        total_bounds_df.drop("filename", axis=1, inplace=True)
    return total_bounds_df
