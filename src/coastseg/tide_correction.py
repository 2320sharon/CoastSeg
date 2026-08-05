"""CoastSeg Tide Correction

Applies tide correction to a session's shoreline time series, locates the FES
tide model on disk, and caches the model datasets that are opened.

Two on-disk model layouts are supported: the un-clipped layout (the default) and
the legacy clipped region0 through region10 layout, which is kept so that
existing installations continue to work.

Tides are predicted in coastseg.tide_predict and read from user supplied CSV
files in coastseg.tide_inputs. Both are re-exported here.
"""

# Standard library imports
import atexit
import logging
import os
import pathlib
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyTMD.io

from coastseg import common, core_utilities, file_utilities
from coastseg.file_utilities import progress_bar_context

# A few names moved out of this module when it was split but are still imported
# from here by the scripts and notebooks, so they stay importable.
from coastseg.tide_inputs import read_content_csv
from coastseg.tide_predict import (
    load_regions_from_geojson,  # noqa: F401  re-exported: scripts import it here
    model_tides,  # noqa: F401  re-exported: scripts import it here
    perform_spatial_join,  # noqa: F401  re-exported: scripts import it here
    predict_tides,
    predict_tides_for_df,  # noqa: F401  re-exported: scripts import it here
)

# Logger setup
logger = logging.getLogger(__name__)

# On-disk layout of each supported model. folders is in preference order.
_MODEL_LAYOUTS: Dict[str, dict] = {
    "FES2022": {
        "folders": ("fes2022b", "fes2022"),
        "format": "FES-netcdf",
        "version": "FES2022",
        "units": "cm",
        "variable": "tide_ocean",
        "n_constituents": 34,
    },
    "FES2014": {
        "folders": ("fes2014",),
        "format": "FES-netcdf",
        "version": "FES2014",
        "units": "cm",
        "variable": "tide_ocean",
        "n_constituents": 34,
    },
}

# Use a consistent aggregation when collapsing duplicate (date, transect_id) records.
# Mean preserves identical tide values while averaging repeated cross-distance measurements.
TIDE_MATRIX_AGGFUNC = "mean"

# pyTMD 2's interpolate.bilinear renormalised over whichever corners of a grid
# cell were wet (sum(w[wet]*z[wet]) / sum(w[wet])). pyTMD 3 interpolates with
# xarray, which returns NaN if any corner is dry; those points then fall through to nearest-neighbour extrapolation instead.
# Enable so tide predictions reproduce pyTMD 2.1.8 partial-cell interpolation at coastlines where some corners of the grid cell are dry instead of returning NaNs or using nearest-neighbour extrapolation ( these tide predictions could be over 10km away and not accurate)
TIDE_PARTIAL_CELL_INTERP = True

# Set to False to predict the ocean tide alone, e.g. to compare against a tide gauge, which is bolted to the crust and so does not sense the load tide.
TIDE_INCLUDE_LOAD_TIDE = True

# How many opened model datasets to keep. The un-clipped model is a single dataset
# shared by every site; the legacy clipped layout needs one per region visited.
TIDE_DATASET_CACHE_SIZE = 4

_DATASET_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_DATASET_CACHE_LOCK = threading.RLock()


# Raw time-series filenames, preferred spelling first then the legacy one, keyed
# by whether the merged (long form) file is wanted.
_RAW_TIMESERIES_FILES: Dict[bool, Tuple[str, ...]] = {
    False: (r"^raw_transect_time_series\.csv$", r"^transect_time_series\.csv$"),
    True: (
        r"^raw_transect_time_series_merged\.csv$",
        r"^transect_time_series_merged\.csv$",
    ),
}


def compute_tidal_corrections(
    session_name: str,
    roi_ids: Collection[str],
    beach_slope: Union[float, str],
    reference_elevation: float,
    only_keep_points_on_transects: bool = False,
    model: str = "FES2022",
    tides_file: str = "",
    use_progress_bar: bool = True,
    tide_model_layout: Optional[str] = "auto",
    tide_model_location: str = "",
) -> None:
    """Computes tidal corrections for specified regions of interest (ROIs).

    Args:
        session_name (str): Name of the session.
        roi_ids (Collection[str]): Collection of ROI identifiers.
        beach_slope (Union[float, str]): Beach slope value or path to file containing slopes.
        reference_elevation (float): Reference elevation in meters relative to MSL.
        only_keep_points_on_transects (bool, optional): If True, only keep points on transects. Defaults to False.
        model (str, optional): Tidal model to use ("FES2022", "FES2014", or ""). Defaults to "FES2022".
        tides_file (str, optional): Path to CSV file containing tide data. Defaults to "".
        use_progress_bar (bool, optional): Whether to display progress bar. Defaults to True.
        tide_model_layout (str, optional): 'auto' (the default, un-clipped with a
            fallback to the clipped regions), 'unclipped', or 'clipped'. See
            `correct_all_tides`.
        tide_model_location (str, optional): Folder holding the tide model. Defaults
            to "", meaning CoastSeg/tide_model.

    Returns:
        None

    Note:
        Observations that cannot be corrected no tide, or no usable slope are
        written with a NaN cross_distance rather than omitted, so the corrected CSVs
        line up row-for-row with the raw ones. See `correct_tides`.

    Raises:
        ValueError: If neither model nor tides_file is provided.
        Exception: Whatever the correction raised. Failures are re-raised rather than
            printed and swallowed, so a caller can tell a completed run from a failed
            one catching everything and returning normally made a missing
            load_tide folder, or an ROI outside the model, look like success.
    """
    logger.info(
        f"Computing tides for ROIs {roi_ids} beach_slope: {beach_slope} reference_elevation: {reference_elevation}"
    )

    try:
        correct_all_tides(
            roi_ids,
            session_name,
            reference_elevation,
            beach_slope,
            only_keep_points_on_transects=only_keep_points_on_transects,
            use_progress_bar=use_progress_bar,
            model=model,
            tides_file=tides_file,
            tide_model_layout=tide_model_layout,
            tide_model_location=tide_model_location,
        )
    except Exception as e:
        # Printed for the notebook, then re-raised so the failure is detectable.
        print(f"Tide Model Error \n {e}")
        logger.exception("tidal correction failed for ROIs %s", roi_ids)
        raise
    print("\ntidal corrections completed")


def correct_all_tides(
    roi_ids: Collection[str],
    session_name: str,
    reference_elevation: float,
    beach_slope: Union[float, str],
    only_keep_points_on_transects: bool = False,
    use_progress_bar: bool = True,
    model: str = "FES2022",
    tides_file: str = "",
    tide_model_layout: Optional[str] = "auto",
    tide_model_location: str = "",
) -> None:
    """Corrects tides for all specified regions of interest (ROIs).

    Validates tide model existence, loads model regions, and corrects tides for each ROI.

    Args:
        roi_ids (Collection[str]): IDs of ROIs to correct tides for.
        session_name (str): Name of session containing extracted shorelines.
        reference_elevation (float): Reference elevation for tide correction.
        beach_slope (Union[float, str]): Beach slope for tide correction.
        only_keep_points_on_transects (bool, optional): Whether to keep only points on transects. Defaults to False.
        use_progress_bar (bool, optional): Whether to display progress bar. Defaults to True.
        model (str, optional): Tide model to use. Defaults to "FES2022".
        tides_file (str, optional): Path to tides file. Defaults to "". When set, the
            tides come from the file and the model is not consulted, so it does not
            have to be downloaded.
        tide_model_layout (str, optional): Which on-disk layout of the model to read.
            'auto', the default, uses the un-clipped model and falls back to the
            legacy clipped region0..region10 layout when that is all that is
            installed. 'unclipped' and 'clipped' name one explicitly, and raise if it
            is not.
        tide_model_location (str, optional): Folder holding the tide model. Defaults
            to "", meaning CoastSeg/tide_model. Point it at the folder that holds
            region0..region10 when a clipped model was built somewhere else.

    Returns:
        None

    Raises:
        ValueError: If neither model nor tides_file is provided. Checked here rather
            than in compute_tidal_corrections because coastseg_map calls this
            directly, so a guard up there never ran on the notebook/UI path.
        FileNotFoundError: If tide_model_layout names a layout that is not installed.
    """
    if model == "" and tides_file == "":
        raise ValueError(
            "Cannot correct tides\nEither set model='FES2014'/model='FES2022' or a provide a file containing tides"
        )

    tide_model_layout = normalize_tide_model_layout(tide_model_layout)

    model_location = ""
    layout = "auto"
    # Validate the tide model only when no tides_file is provided.
    # A supplied tides_file bypasses the model entirely.
    if model != "" and tides_file == "":
        # Located and resolved in one pass, so the folder and the layout cannot
        # disagree, and threaded down so the probe does not repeat per ROI.
        model_location, layout = locate_tide_model(
            tide_model_location,
            model=model.lower(),
            tide_model_layout=tide_model_layout,
        )
        print(
            f"Using the {model} model at {model_location} "
            f"({'un-clipped' if layout == 'global' else 'legacy clipped regions'} layout)."
        )
        logger.info("tide model %s layout=%s", model_location, layout)

    # One source for the whole run: the model is set up, and a tides file checked
    # for, once rather than per ROI.
    tide_source = _build_tide_source(
        tides_file=tides_file,
        model=model,
        model_location=model_location,
        tide_model_layout=layout,
    )

    with progress_bar_context(
        use_progress_bar,
        total=len(roi_ids),
        description=f"Correcting Tides for {len(roi_ids)} ROIs",
    ) as update:
        for roi_id in roi_ids:
            correct_tides(
                roi_id,
                session_name,
                reference_elevation,
                beach_slope,
                only_keep_points_on_transects=only_keep_points_on_transects,
                use_progress_bar=use_progress_bar,
                tides_file=tides_file,
                model=model,
                model_location=model_location,
                tide_model_layout=layout,
                tide_source=tide_source,
            )
            logger.info(f"{roi_id} was tidally corrected")
            update(f"{roi_id} was tidally corrected")


def save_transect_settings(
    session_path: str,
    reference_elevation: float,
    beach_slope: float,
    filename: str = "transects_settings.json",
    *,
    tide_model: Optional[str] = None,
    tide_model_layout: Optional[str] = None,
    tide_model_location: Optional[str] = None,
) -> None:
    """Updates and saves transect settings with provided reference elevation and beach slope.

    Args:
        session_path (str): Path to session directory containing the transect settings JSON file.
        reference_elevation (float): Reference elevation value to update.
        beach_slope (float): Beach slope value to update.
        filename (str, optional): Name of the JSON settings file. Defaults to "transects_settings.json".
        tide_model (str, optional): Tide model the session was corrected with. Only
            written when supplied.
        tide_model_layout (str, optional): 'unclipped' or 'clipped', the layout the
            tides were read from. Recorded in the user-facing spelling so it can be
            handed straight back to compute_tidal_corrections. Only written when
            supplied.
        tide_model_location (str, optional): Folder the model was read from. Only
            written when supplied.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified settings file does not exist.
    """
    transects_settings = file_utilities.read_json_file(
        os.path.join(session_path, filename), raise_error=False
    )
    transects_settings["reference_elevation"] = reference_elevation
    transects_settings["beach_slope"] = beach_slope
    for key, value in (
        ("tide_model", tide_model),
        ("tide_model_layout", tide_model_layout),
        ("tide_model_location", tide_model_location),
    ):
        if value:
            transects_settings[key] = value
    file_utilities.to_file(transects_settings, os.path.join(session_path, filename))


def apply_tide_correction_df(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Applies tide correction to timeseries DataFrame by adjusting cross-shore distances.

    Calculates tidal corrections based on tide-reference elevation difference
    normalized by beach slope, then applies to cross-shore distance measurements.

    Args:
        timeseries (pd.DataFrame): DataFrame with required columns:
            - 'tide': Tide level at each time point.
            - 'reference_elevation': Reference elevation to compare against.
            - 'slope': Beach slope.
            - 'cross_distance': Original cross-shore distance to adjust.

    Returns:
        pd.DataFrame: Modified DataFrame with 'cross_distance' adjusted for tide correction.
                     Temporary 'correction' column is removed.

    Raises:
        KeyError: If any required columns are missing.

    Example:
        >>> df = pd.DataFrame(
        ...     {
        ...         "tide": [1.2, 0.8],
        ...         "reference_elevation": [0.0, 0.0],
        ...         "slope": [0.1, 0.1],
        ...         "cross_distance": [50.0, 45.0],
        ...     }
        ... )
        >>> result = apply_tide_correction_df(df)
        >>> print(result)
           tide  reference_elevation  slope  cross_distance
        0   1.2                  0.0    0.1            62.0
        1   0.8                  0.0    0.1            53.0

    Notes:
        Correction formula: correction = (tide - reference_elevation) / slope
        Adjusted distance: cross_distance + correction

        The division is unguarded, as it has always been. A missing tide or slope
        gives a NaN cross_distance; a slope of 0 gives an infinite one, because the
        slope converts a vertical tide offset into a horizontal displacement and a
        flat beach has no finite answer.
    """
    reference_elevation = timeseries["reference_elevation"]
    beach_slope = timeseries["slope"]
    timeseries["correction"] = (timeseries["tide"] - reference_elevation) / beach_slope
    timeseries["cross_distance"] = (
        timeseries["cross_distance"] + timeseries["correction"]
    )
    # drop correction
    timeseries.drop(columns=["correction"], inplace=True)
    return timeseries


def _pivot_matrix(
    df: pd.DataFrame,
    values: str,
    aggfunc: Optional[str] = None,
    stringify_ids: bool = True,
) -> pd.DataFrame:
    """Pivot a long time series into a dates x transect_id matrix.

    Duplicate observations for the same date and transect are aggregated using
    aggfunc. If no aggregation function is provided, TIDE_MATRIX_AGGFUNC is used.

    Args:
        df (pd.DataFrame): Long-form frame with 'dates', 'transect_id' and values.
        values (str): Column containing the values to populate the matrix.
        aggfunc (str, optional): Aggregation function used for duplicate date-transect pairs.
            Defaults to TIDE_MATRIX_AGGFUNC.
        stringify_ids (bool, optional): Whether to convert transect IDs to strings before using
            them as column labels. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with "dates" as the first column and one column per
        transect ID.

    Note:
        A matrix cannot represent a transect that intersected the shoreline more
        than once on the same date, so those rows are collapsed and a warning is
        logged. The long-form *_merged.csv keeps every observation.

        dropna=False preserves transects whose values are entirely missing as
        all-NaN columns.
    """
    matrix = df.copy()
    if stringify_ids:
        matrix["transect_id"] = matrix["transect_id"].astype(str)

    duplicated = int(matrix.duplicated(subset=["dates", "transect_id"]).sum())
    if duplicated:
        logger.warning(
            "%d observation(s) share a (date, transect_id) and were combined with "
            "'%s' to fit the %s matrix; the merged CSV keeps them all",
            duplicated,
            aggfunc or TIDE_MATRIX_AGGFUNC,
            values,
        )

    matrix = matrix.pivot_table(
        index="dates",
        columns="transect_id",
        values=values,
        aggfunc=aggfunc or TIDE_MATRIX_AGGFUNC,
        dropna=False,
    )
    # Reset index if you want 'dates' back as a column
    return matrix.reset_index()


def save_predicted_tides_to_csv(
    session_path: str, predicted_tides_df: pd.DataFrame
) -> pd.DataFrame:
    """Saves the predicted tides DataFrame to a CSV file after pivoting it.

    Args:
        session_path (str): The directory path where the CSV file will be saved.
        predicted_tides_df (pd.DataFrame): DataFrame containing predicted tides with columns 'dates', 'transect_id', and 'tide'.

    Returns:
        pd.DataFrame: The pivoted DataFrame with 'dates' as the index and 'transect_id' as columns.
    """
    # pivot to dates x transect_ids
    pivot_df = _pivot_matrix(predicted_tides_df, "tide", stringify_ids=False)
    pivot_df.to_csv(os.path.join(session_path, "predicted_tides.csv"), index=False)
    return pivot_df


def get_matrix_timeseries(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Returns a timeseries DataFrame as a dates x transect_id matrix.

    This function takes a timeseries DataFrame and pivots it to create a matrix
    where the rows are dates and the columns are transect IDs.

    Args:
        timeseries (pd.DataFrame): The input DataFrame containing the timeseries data.
            It must have columns 'dates', 'transect_id', and 'cross_distance'.

    Returns:
        pd.DataFrame: The pivoted DataFrame (matrix) with 'dates' as a column and
            'transect_id' as the remaining columns.

    Note:
        The transect ids are converted to strings.
    """
    return _pivot_matrix(timeseries, "cross_distance")


def _warn_about_uncorrectable_observations(
    timeseries: pd.DataFrame, roi_id: str
) -> None:
    """Report the observations that will come through the correction as NaN.

    An observation is considered uncorrectable when either its tide or slope
    value is missing. These observations are retained in the output with a NaN
    cross_distance rather than being dropped.

    If all observations for a transect are uncorrectable, the transect is also
    reported because its output column will contain only NaN values.

    Args:
        timeseries: DataFrame containing "transect_id", "tide", and "slope"
            columns.
        roi_id: ROI identifier included in the warning message.

    Returns:
        None
    """
    unusable = timeseries["tide"].isna() | timeseries["slope"].isna()
    if not unusable.any():
        return

    by_transect = unusable.groupby(timeseries["transect_id"].astype(str)).all()
    entirely_lost = sorted(by_transect[by_transect].index)

    logger.warning(
        "%s: %d of %d observation(s) have no usable tide or slope and are written "
        "with a NaN cross_distance rather than dropped%s",
        roi_id,
        int(unusable.sum()),
        len(timeseries),
        (
            f"; {len(entirely_lost)} transect(s) have no usable observation at all "
            f"and come out as an empty column: {entirely_lost[:20]}"
            if entirely_lost
            else ""
        ),
    )


class TideSource(NamedTuple):
    """A source of tide values for a single ROI.

    Tides are either read from a file provided by the user or predicted from a
    tide model. Both are created as a TideSource, so correct_tides does not need
    to know which one it was given. A new kind of source is added by writing
    another builder function such as tides_from_file or tides_from_model.

    Attributes:
        describe (str): Text displayed on the progress bar, for example
            'Predicting tides'.
        resolve (Callable): Called with (timeseries, transects_gdf). Returns the
            time series with a 'tide' column, containing NaN where no tide value
            was found.
    """

    describe: str
    resolve: Callable[[pd.DataFrame, gpd.GeoDataFrame], pd.DataFrame]


def tides_from_file(tides_file: str) -> TideSource:
    """Creates a TideSource that reads tide values from a CSV file.

    The tide model is not used, so it does not need to be downloaded. The
    supported CSV formats are handled by coastseg.tide_inputs.read_content_csv.

    Args:
        tides_file (str): Path to the CSV file containing tides.

    Returns:
        TideSource: A source that reads tides from the specified file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(tides_file):
        raise FileNotFoundError(f"Tide CSV file not found at {tides_file}")

    def resolve(
        timeseries: pd.DataFrame, transects_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        return read_content_csv(tides_file, timeseries, column_name="tide")

    return TideSource("Reading tides from file", resolve)


def tides_from_model(
    model: str = "FES2022",
    model_location: str = "",
    tide_model_layout: Optional[str] = "auto",
    tide_regions_file: str = "",
) -> TideSource:
    """Creates a TideSource that predicts tide values from a tide model.

    Tides are predicted at the seaward point of each transect.

    Args:
        model (str, optional): Tide model to use. Defaults to "FES2022".
        model_location (str, optional): Path to the tide model. Defaults to "".
        tide_model_layout (str, optional): Which layout to read: 'auto' (the
            default), 'unclipped' or 'clipped'. This is normally the layout that
            correct_all_tides already resolved, so the model directory is not
            searched again for each ROI.
        tide_regions_file (str, optional): Path to the file containing the regions
            the tide model was clipped to. Only used by the clipped layout.
            Defaults to the file included with CoastSeg.

    Returns:
        TideSource: A source that predicts tides from the specified model.
    """
    config = setup_tide_model_config(
        model_location, model=model, tide_model_layout=tide_model_layout
    )
    regions_file = tide_regions_file
    if not regions_file and config["LAYOUT"] == "regions":
        regions_file = file_utilities.load_package_resource(
            "tide_model", "tide_regions_map.geojson"
        )

    def resolve(
        timeseries: pd.DataFrame, transects_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        tides = predict_tides(
            transects_gdf, _observation_dates(timeseries), regions_file, config
        )

        # Match transect ID types before merging.
        tides["transect_id"] = tides["transect_id"].astype(str)
        observations = timeseries.assign(
            transect_id=timeseries["transect_id"].astype(str)
        )

        # the tide's own x/y is the transect's seaward point, not the shoreline
        # intersection the time series carries under the same names
        tides = tides.drop(columns=["x", "y"], errors="ignore")

        # Left-merge onto the time series so every observation survives. Any transect
        # without tides has NaN values
        return observations.merge(tides, on=["transect_id", "dates"], how="left")

    return TideSource("Predicting tides", resolve)


def _observation_dates(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Returns the transect ID and date pairs that require a tide value.

    Each pair appears only once, otherwise the predicted tides would duplicate
    observations when they are merged back onto the time series. Observations
    without a cross_distance are excluded because there is nothing to correct.

    Args:
        timeseries (pd.DataFrame): The raw merged time series containing
            'transect_id', 'dates' and 'cross_distance' columns.

    Returns:
        pd.DataFrame: One row per transect ID and date that requires a tide.
    """
    wanted = timeseries[timeseries["cross_distance"].notna()]
    return wanted[["transect_id", "dates"]].drop_duplicates()


def _build_tide_source(
    tides_file: str = "",
    model: str = "FES2022",
    model_location: str = "",
    tide_regions_file: str = "",
    tide_model_layout: Optional[str] = "auto",
) -> TideSource:
    """Creates a TideSource from the provided tide settings.

    A tides_file takes priority over the tide model.

    Args:
        tides_file (str, optional): Path to a CSV file containing tides, or "" to
            use the tide model instead. Defaults to "".
        model (str, optional): Tide model to use. Defaults to "FES2022".
        model_location (str, optional): Path to the tide model. Defaults to "".
        tide_regions_file (str, optional): Path to the file containing the regions
            the tide model was clipped to. Only used by the clipped layout.
        tide_model_layout (str, optional): Which layout to read: 'auto' (the
            default), 'unclipped' or 'clipped'.

    Returns:
        TideSource: The source to read the ROI's tide values from.
    """
    if tides_file:
        return tides_from_file(tides_file)
    return tides_from_model(
        model=model,
        model_location=model_location,
        tide_model_layout=tide_model_layout,
        tide_regions_file=tide_regions_file,
    )


def _resolve_tides(
    timeseries: pd.DataFrame,
    transects_gdf: gpd.GeoDataFrame,
    *,
    roi_id: str,
    tides_file: str = "",
    model: str = "FES2022",
    model_location: str = "",
    tide_regions_file: str = "",
    tide_model_layout: Optional[str] = "auto",
    tide_source: Optional[TideSource] = None,
    update: Callable[[str], None] = lambda message: None,
) -> pd.DataFrame:
    """Add tide values to a transect time series.

    Where the tides come from is the TideSource. Pass one, or pass the keyword
    arguments below and one is built for you, which is what every caller did
    before the seam existed.

    Args:
        timeseries (pd.DataFrame): The raw merged time series, long form.
        transects_gdf (gpd.GeoDataFrame): Transects for this ROI, used to place the
            seaward point each tide is predicted at. Unused when reading a file.
        roi_id (str): ROI identifier, for the progress messages.
        tides_file (str): Path to a tide CSV. When set, the model is not used.
        model (str): Tide model name, e.g. 'FES2022'.
        model_location (str): Folder holding the tide model.
        tide_regions_file (str): Path to LEGACY tide region map file, only read on the clipped layout.
        tide_model_layout (str): 'auto', 'unclipped' or 'clipped'. Normally the layout
            correct_all_tides already resolved, so it is not resolved again per ROI.
        tide_source (TideSource, optional): Where to get the tides. Built from the
            arguments above when not given, and used instead of them when given.
        update (Callable[[str], None]): Progress reporter.

    Returns:
        pd.DataFrame: The time series with a "tide" column. Observations without a matching
        or predicted tide value contain NaN.

    Raises:
        FileNotFoundError: If tides_file is provided but does not exist.
    """
    if tide_source is None:
        tide_source = _build_tide_source(
            tides_file=tides_file,
            model=model,
            model_location=model_location,
            tide_regions_file=tide_regions_file,
            tide_model_layout=tide_model_layout,
        )

    update(f"{tide_source.describe} : {roi_id}")
    return tide_source.resolve(timeseries, transects_gdf)


def _attach_slopes(
    timeseries: pd.DataFrame, beach_slope: Union[float, str]
) -> pd.DataFrame:
    """Attach a 'slope' column, either a single value or one matched from a CSV.

    Args:
        timeseries (pd.DataFrame): The time series to add the column to.
        beach_slope (float | str): A slope for every observation, or the path to a
            slope CSV in any of the documented layouts.

    Returns:
        pd.DataFrame: The time series with a 'slope' column.
    """
    if isinstance(beach_slope, str):
        timeseries = timeseries.assign(slope=np.nan)
        return read_content_csv(beach_slope, timeseries, column_name="slope")
    return timeseries.assign(slope=beach_slope)


def _save_corrected_outputs(
    corrected: pd.DataFrame,
    transects_gdf: gpd.GeoDataFrame,
    session_path: str,
    only_keep_points_on_transects: bool = False,
) -> pd.DataFrame:
    """Save tide-corrected time-series outputs.

    Saves the predicted tides, corrected point and vector GeoJSON files,
    corrected matrix CSV, and corrected long-form CSV.

    Writes, in order:

    1 predicted_tides.csv : dates x transect_id matrix of the tides used
    2 tidally_corrected_transect_time_series_points.geojson and _vectors.geojson
    3 tidally_corrected_transect_time_series.csv : the corrected matrix
    4 tidally_corrected_transect_time_series_merged.csv : the corrected long form

    Args:
        corrected (pd.DataFrame): The corrected long-form time series.
        transects_gdf (gpd.GeoDataFrame): Transects for this ROI, in any CRS.
        session_path (str): Directory where output files are saved.
        only_keep_points_on_transects (bool): Drop shoreline points that do not fall
            on their transect, recording them in dropped_points_time_series.csv.

    Returns:
        pd.DataFrame: The corrected long-form time series, with shore_x/shore_y added.
    """
    # Save the tide values used for correction.
    save_predicted_tides_to_csv(session_path, corrected)

    corrected_matrix = get_matrix_timeseries(corrected)

    merged_df, matrix_df = common.add_lat_lon_to_timeseries(
        corrected,
        transects_gdf.to_crs("epsg:4326"),
        corrected_matrix,
        session_path,
        only_keep_points_on_transects,
        "tidally_corrected",
    )

    matrix_df = matrix_df[common.sort_matrix_columns(matrix_df.columns)]
    matrix_df.to_csv(
        os.path.join(session_path, "tidally_corrected_transect_time_series.csv"),
        index=False,
    )
    merged_df.to_csv(
        os.path.join(session_path, "tidally_corrected_transect_time_series_merged.csv"),
        index=False,
    )
    return merged_df


def correct_tides(
    roi_id: str,
    session_name: str,
    reference_elevation: float,
    beach_slope: Union[float, str],
    only_keep_points_on_transects: bool = False,
    use_progress_bar: bool = True,
    tides_file: str = "",
    model: str = "FES2022",
    tide_regions_file: str = "",
    model_location: str = "",
    tide_model_layout: Optional[str] = "auto",
    tide_source: Optional[TideSource] = None,
) -> pd.DataFrame:
    """Apply tide correction to a transect time series.

    Tide values are read from a CSV file when tides_file is provided;
    otherwise, they are predicted from the selected tide model. Beach slopes
    may be provided as a constant value or loaded from a file.

    Observations without a usable tide or slope are retained with NaN
    cross_distance values rather than being dropped.

    Args:
        roi_id (str): Identifier for the Region Of Interest.
        session_name (str): Name of the session.
        reference_elevation (float): Reference elevation value.
        beach_slope (Union[float, str]): Slope of the beach, or the path to a file containing slopes.
        only_keep_points_on_transects (bool, optional): If True, keeps only the shoreline
            points that are on the transects. Defaults to False.
            - This will generate a file called "dropped_points_time_series.csv" that contains the points that were filtered out.
            - Any shoreline points that were not on the transects will be removed from "raw_transect_time_series.csv" by setting those values to NaN.
            - The "raw_transect_time_series_merged.csv" will not contain any points that were not on the transects.
        use_progress_bar (bool, optional): If True, a tqdm progress bar will be displayed. Defaults to True.
        tides_file (str, optional): Path to a CSV file containing tides. Defaults to "".
        model (str, optional): Tide model to use. Defaults to "FES2022".
        tide_regions_file (str, optional): Path to the file containing the regions the
            tide model was clipped to. Defaults to "". Ignored when tide_source is
            given; see Note.
        model_location (str, optional): Path to the tide model. Defaults to "".
        tide_model_layout (str, optional): Which layout to read: 'auto' (the default),
            'unclipped' or 'clipped'.
        tide_source (TideSource, optional): Where to get this ROI's tides, taking
            precedence over the arguments above (see Note). Built from them when not
            given; correct_all_tides builds one for the whole run so the model is set
            up once rather than per ROI.

    Returns:
        pd.DataFrame: Tide-corrected long-form time-series data for the specified ROI.
        Returns an empty DataFrame when no time-series data is available.
    """
    with progress_bar_context(use_progress_bar, total=6) as update:
        update(f"Getting time series for ROI : {roi_id}")
        # read the merged csv of the raw timeseries (aka not tidally corrected)
        try:
            timeseries = get_timeseries(roi_id, session_name, is_merged=True)
        except FileNotFoundError:
            timeseries = pd.DataFrame()

        # Skip correction when no shoreline observations are available.
        if timeseries.empty:
            message = (
                f"No time series data found for {roi_id} cannot perform tide correction"
            )
            print(message)
            logger.warning(message)
            update(message)
            return pd.DataFrame()

        session_path = file_utilities.get_session_contents_location(
            session_name, roi_id
        )
        # File-based slopes have no single value to save.
        saved_slope = np.nan if isinstance(beach_slope, str) else beach_slope
        save_transect_settings(session_path, reference_elevation, saved_slope)
        # read the transects from the config_gdf.geojson file
        update(f"Getting transects for ROI : {roi_id}")
        transects_gdf = get_transects(roi_id, session_name)

        timeseries = _resolve_tides(
            timeseries,
            transects_gdf,
            roi_id=roi_id,
            tides_file=tides_file,
            model=model,
            model_location=model_location,
            tide_regions_file=tide_regions_file,
            tide_model_layout=tide_model_layout,
            tide_source=tide_source,
            update=update,
        )

        # Now that the tides resolved, record which model they came from.
        if model and not tides_file:
            save_transect_settings(
                session_path,
                reference_elevation,
                saved_slope,
                tide_model=model,
                tide_model_layout=_LAYOUT_REPORTED.get(
                    normalize_tide_model_layout(tide_model_layout)
                ),
                tide_model_location=portable_model_location(model_location),
            )

        timeseries = _attach_slopes(timeseries, beach_slope)
        _warn_about_uncorrectable_observations(timeseries, roi_id)
        timeseries["reference_elevation"] = reference_elevation

        update(f"Tidally correcting time series for ROI : {roi_id}")
        # assumes timeseries contains the columns tide, reference_elevation and slope
        corrected = apply_tide_correction_df(timeseries)

        update(f"Saving tidally corrected time series for ROI : {roi_id}")
        merged_df = _save_corrected_outputs(
            corrected,
            transects_gdf,
            session_path,
            only_keep_points_on_transects,
        )

        update(f"{roi_id} was tidally corrected")
    return merged_df


def get_timeseries(
    ROI_ID: str, session_name: str, is_merged: bool = False
) -> pd.DataFrame:
    """Retrieves the raw timeseries DataFrame for a given ROI ID and session name.

    Args:
        ROI_ID (str): ID of the region of interest.
        session_name (str): Name of the session.
        is_merged (bool, optional): If True, retrieves merged timeseries file. Defaults to False.

    Returns:
        pd.DataFrame: Raw timeseries DataFrame containing shoreline data.

    Raises:
        FileNotFoundError: If the timeseries file is not found for the specified ROI and session.
    """
    session_path = file_utilities.get_session_contents_location(session_name, ROI_ID)
    for pattern in _RAW_TIMESERIES_FILES[bool(is_merged)]:
        try:
            path = file_utilities.find_file_by_regex(session_path, pattern)
        except FileNotFoundError:
            continue
        return timeseries_read_csv(path, is_merged)
    raise FileNotFoundError(
        f"No raw time series for {ROI_ID} in {session_path}. Looked for "
        f"{list(_RAW_TIMESERIES_FILES[bool(is_merged)])}."
    )


def get_transects(roi_id: str, session_name: str) -> gpd.GeoDataFrame:
    """Retrieves the transects GeoDataFrame for a specific ROI and session.

    Locates and reads the transects configuration file (config_gdf.geojson)
    for the specified region of interest, filtering for transect features only.

    Args:
        roi_id (str): Identifier for the region of interest.
        session_name (str): Name of the session containing the ROI data.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing transect geometries and metadata for the specified ROI.

    Raises:
        FileNotFoundError: If the configuration file is not found.

    Example:
        >>> transects_gdf = get_transects("roi_001", "my_session")
        >>> print(transects_gdf.columns.tolist())
        ['id', 'type', 'geometry']
        >>> print(len(transects_gdf))
        25  # Number of transects for this ROI
    """
    # open the sessions directory
    session_path = file_utilities.get_session_location(session_name)
    roi_location = file_utilities.find_matching_directory_by_id(session_path, roi_id)
    if roi_location is not None:
        session_path = roi_location
    # locate the config_gdf.geojson containing the transects
    config_path = file_utilities.find_file_by_regex(
        session_path, r"^config_gdf\.geojson$"
    )
    # Load the GeoJSON file containing transect data
    transects_gdf = read_and_filter_geojson(config_path, feature_type="transect")
    return transects_gdf


def setup_tide_model_config(
    model_path: str, model: str, tide_model_layout: Optional[str] = "auto"
) -> Dict[str, Union[str, list, bool, float, int]]:
    """Set up configuration dictionary for tide model computations.

    This function creates a standardized configuration dictionary containing
    all the parameters needed for tide model predictions using pyTMD.

    Args:
        model_path (str): Path to the directory containing the tide model data files.
        model (str): Name of the tide model in uppercase (e.g., 'FES2022', 'FES2014').
        tide_model_layout (str, optional): Which layout to read: 'auto' (the default,
            resolved from disk), 'unclipped' or 'clipped'.

    Returns:
        Dict[str, Union[str, list, bool, float, int]]: Configuration dictionary containing:
            - DIRECTORY: Path to model data
            - MODEL: Model name in uppercase
            - EXTRAPOLATE: Whether to extrapolate beyond model domain
            - METHOD: Interpolation method ('bilinear')
            - EPSG: Coordinate system (4326)
            - CUTOFF: Extrapolation cutoff distance
            - REGION_DIRECTORY: LEGACY, path prefix for the clipped region folders
            - LAYOUT: 'global' for the un-clipped model (the default), or 'regions'
              for the legacy clipped layout, which may be removed in a future
              version of CoastSeg

    Example:
        >>> config = setup_tide_model_config("/path/to/model", "FES2022")
        >>> print(config["MODEL"])
        'FES2022'
        >>> print(config["EXTRAPOLATE"])
        True
    """
    return {
        "DIRECTORY": model_path,
        "MODEL": model.upper(),  # name of the model in uppercase eg FES2022
        "EXTRAPOLATE": True,
        "METHOD": "bilinear",  # pyTMD 3 maps 'bilinear' onto its 'linear' interpolation
        "EPSG": 4326,
        "CUTOFF": 10,
        # LEGACY: only consumed when LAYOUT == 'regions'
        "REGION_DIRECTORY": os.path.join(model_path, "region"),
        "LAYOUT": resolve_model_layout(
            model_path, model, tide_model_layout=tide_model_layout
        ),
    }


def required_tide_groups() -> Tuple[str, ...]:
    """The constituent groups a prediction will actually open.

    TIDE_INCLUDE_LOAD_TIDE decides whether load_tide is needed, so validation
    and prediction read the requirement from the same place.

    Returns:
        Tuple[str, ...]: 'ocean_tide', plus 'load_tide' when the load tide is on.
    """
    return ("ocean_tide", "load_tide") if TIDE_INCLUDE_LOAD_TIDE else ("ocean_tide",)


def _load_tide_missing_error(cause: Exception) -> FileNotFoundError:
    """The actionable error for a model that has no load_tide folder.

    Args:
        cause (Exception): The underlying lookup failure, quoted in the message.

    Returns:
        FileNotFoundError: Ready to raise.
    """
    return FileNotFoundError(
        f"{cause}\n\nThe load tide is required to match CoastSat's tide levels, "
        "which report the geocentric tide (ocean + load). Download the "
        "load_tide folder alongside ocean_tide, or set "
        "coastseg.tide_correction.TIDE_INCLUDE_LOAD_TIDE = False to predict "
        "the ocean tide only."
    )


def _check_group_is_complete(
    directory: Union[str, pathlib.Path], model_key: str, group: str, definition: dict
) -> None:
    """Refuse a constituent group that is short of the files the model declares.

    A short group does not fail loudly on its own. pyTMD reads whatever is there and
    predicts from it, so an interrupted download produces a tide quietly wrong by
    centimeters or, once enough majors have landed for minor-constituent inference
    to start reaching for one that has not, a bare KeyError: 'q1' from inside
    pyTMD that says nothing about the real problem.

    Args:
        directory (str | pathlib.Path): The tide model folder, for the message.
        model_key (str): Canonical model name, a key of _MODEL_LAYOUTS.
        group (str): The constituent group being checked.
        definition (dict): The definition built for that group.

    Raises:
        FileNotFoundError: If the group declares fewer files than the model should
            have. A surplus is left alone `build_model_definition` already
            warns about it, and it is a different problem.
    """
    expected = int(_MODEL_LAYOUTS[model_key]["n_constituents"])
    found = len(definition["z"]["model_file"])
    if found < expected:
        raise FileNotFoundError(
            f"The {model_key} '{group}' group under {directory} holds {found} of the "
            f"{expected} constituent files, so the tide model is incomplete. "
            "Re-run the download to fetch what is missing the files that "
            "already finished are skipped."
        )


def require_tide_groups(
    directory: Union[str, pathlib.Path], model: str
) -> Tuple[str, ...]:
    """Fail now, with guidance, if a group the prediction needs is missing from disk.

    Both halves of "missing" are checked: a group with no folder at all, and a group
    whose folder is there but short of constituents.

    Args:
        directory (str | pathlib.Path): The tide model folder.
        model (str): Tide model name in any accepted spelling.

    Returns:
        Tuple[str, ...]: The groups that were checked and resolved.

    Raises:
        FileNotFoundError: If a required group's constituent files are missing. The
            load_tide case carries the same guidance `model_tides` gives, so
            the advice does not depend on where the gap was noticed.
    """
    groups = required_tide_groups()
    key = normalize_model_name(model)
    for group in groups:
        try:
            definition = build_model_definition(directory, model, group=group)
        except (FileNotFoundError, ValueError) as exc:
            if group == "load_tide":
                raise _load_tide_missing_error(exc) from exc
            raise
        _check_group_is_complete(directory, key, group, definition)
    return groups


# User-facing spellings of the layout choice, mapped onto the internal names.
_LAYOUT_ALIASES = {
    "": "auto",
    "auto": "auto",
    "unclipped": "global",
    "global": "global",
    "clipped": "regions",
    "regions": "regions",
}

# The spelling to write into a session's transects_settings.json: the user-facing one,
# so a recorded value can be handed straight back to compute_tidal_corrections. 'auto'
# maps to None because it names no particular layout, so there is nothing to record.
_LAYOUT_REPORTED = {"global": "unclipped", "regions": "clipped", "auto": None}


def portable_model_location(model_location: str) -> str:
    """Store the tide model folder relative to the CoastSeg directory when it is inside it.

    Session folders get copied between machines, so an absolute path baked into one is
    dead everywhere else and carries the running user's home directory with it. Same
    reasoning as common.relativize_sar_model_path.

    Args:
        model_location (str): The folder the model was read from.

    Returns:
        str: A POSIX path, relative to the CoastSeg directory where possible. Empty
        input returns empty, which save_transect_settings then does not record.
    """
    if not model_location:
        return ""
    candidate = pathlib.Path(model_location)
    try:
        return candidate.relative_to(core_utilities.get_base_dir()).as_posix()
    except ValueError:
        return candidate.as_posix()


def normalize_tide_model_layout(tide_model_layout: Optional[str] = "auto") -> str:
    """Map any spelling of the layout choice onto its internal name.

    Accepts the user-facing spellings ('unclipped', 'clipped'), the internal ones
    ('global', 'regions'), and hyphenated/underscored variants. None and the empty
    string mean 'auto', so an unset argument behaves as it always did.

    Args:
        tide_model_layout (str, optional): The layout in any accepted spelling.

    Returns:
        str: 'auto', 'global' or 'regions'.

    Raises:
        ValueError: If the layout is not one CoastSeg supports. A typo must not
        mean 'auto'.

    Example:
        >>> normalize_tide_model_layout("un-clipped")
        'global'
    """
    if tide_model_layout is None:
        return "auto"
    key = str(tide_model_layout).strip().lower().replace("-", "").replace("_", "")
    try:
        return _LAYOUT_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unsupported tide model layout {tide_model_layout!r}; expected one of "
            "['auto', 'clipped', 'unclipped'] (the internal spellings 'global' and "
            "'regions' are also accepted)."
        ) from None


#: The legacy clipped layout is always region0 through region10.
_N_TIDE_REGIONS = 11


def resolve_model_layout(
    tide_model_root: Union[str, pathlib.Path],
    model: str,
    tide_model_layout: Optional[str] = "auto",
) -> str:
    """Determine whether to use the global or legacy regional tide model layout.

    A named layout is returned as given. 'auto' prefers the global, un-clipped model
    and falls back to the legacy clipped region layout when that is all that is on
    disk, so an install that only ever ran the clipping step keeps working with no
    configuration.

    Args:
        tide_model_root (str | pathlib.Path): The tide_model folder. Keep this the
            first positional parameter, because tests/conftest.py's tide-model
            guard reads args[0] to decide whether a test touches the real model.
        model (str): Tide model name in any accepted spelling.
        tide_model_layout (str, optional): 'auto' (the default), 'unclipped' or
            'clipped'. An explicit choice is returned without reading the disk. Use
            require_model_layout to check it is actually installed.

    Returns:
        str:  "global" when the un-clipped model is used, otherwise "regions".

    Note:
        The region0 through region10 layout is deprecated and may be removed
        in a future CoastSeg release. It remains supported for compatibility
        with existing installations.
    """
    choice = normalize_tide_model_layout(tide_model_layout)
    if choice != "auto":
        return choice

    if _model_readable(tide_model_root, model):
        return "global"

    logger.info(
        "no un-clipped %s model under %s; falling back to the legacy clipped "
        "region0..region10 layout. Support for the clipped layout may be "
        "removed in a future version of CoastSeg re-downloading the model "
        "without running the clipping step is faster and covers the whole "
        "planet.",
        model,
        tide_model_root,
    )
    return "regions"


def _model_readable(directory: Union[str, pathlib.Path], model: str) -> bool:
    """Checks whether a tide model can be read from the specified directory.

    Used to determine which layout is installed. Point it at the tide model
    directory for the un-clipped layout, or at a single region directory for the
    clipped layout.

    Args:
        directory (str | pathlib.Path): The directory to check.
        model (str): The tide model to check for, in any accepted spelling.

    Returns:
        bool: True if a model definition can be built from the directory. An
            unsupported model name returns False instead of raising an error.
    """
    try:
        build_model_definition(directory, model)
    except (FileNotFoundError, ValueError):
        return False
    return True


def require_model_layout(
    tide_model_root: Union[str, pathlib.Path], model: str, tide_model_layout: str
) -> None:
    """Refuse an explicitly requested layout that is not installed.

    Args:
        tide_model_root (str | pathlib.Path): The folder the model was asked for in.
        model (str): Tide model name in any accepted spelling.
        tide_model_layout (str): 'auto', 'unclipped'/'global' or 'clipped'/'regions'.
            'auto' is a no-op, because the auto path is allowed to fall back.

    Returns:
        None

    Raises:
        FileNotFoundError: If the requested layout is not installed at that root.
        ValueError: If model is not a tide model CoastSeg supports.
    """
    layout = normalize_tide_model_layout(tide_model_layout)
    if layout == "auto":
        return

    key = normalize_model_name(model)
    root = pathlib.Path(tide_model_root)
    folders = _MODEL_LAYOUTS[key]["folders"]

    if layout == "global":
        if _model_readable(root, model):
            return
        regions_at = _find_clipped_regions(root, key, model)
        if regions_at == root:
            hint = (
                "The legacy clipped region0..region10 layout IS present there. Set "
                "tide_model_layout='clipped' (or 'auto') to use it, or re-run "
                "Download_tide_model.ipynb it fetches the un-clipped model, and "
                "the clipping step is no longer needed."
            )
        elif regions_at is not None:
            hint = (
                f"The legacy clipped region0..region10 layout IS present, at "
                f"{regions_at}. Set tide_model_layout='clipped' and "
                f'tide_model_location=r"{regions_at}" to use it, or re-run '
                f"Download_tide_model.ipynb for the un-clipped model."
            )
        else:
            hint = (
                f"There is no {key} tide model of either layout at {root}. Run "
                f"Download_tide_model.ipynb, and skip its clipping section."
            )
        raise FileNotFoundError(
            f"No un-clipped {key} tide model under {root}. "
            f"tide_model_layout='unclipped' asks for ocean_tide[_<YYYYMMDD>] and "
            f"load_tide, each with {_MODEL_LAYOUTS[key]['n_constituents']} .nc files, "
            f"inside a model folder named one of {list(folders)} directly at that "
            f"path. {hint}"
        )

    # layout == "regions"
    missing = [
        f"region{index}"
        for index in range(_N_TIDE_REGIONS)
        if not _model_readable(root / f"region{index}", model)
    ]
    if not missing:
        return

    found = _find_clipped_regions(root, key, model)
    if found is not None and found != root:
        hint = (
            f"Clipped regions were found at {found}, not at the folder named. The "
            f"model root and the model folder differ by one level and are easy to "
            f'confuse. Pass tide_model_location=r"{found}".'
        )
    elif _model_readable(root, model):
        hint = (
            f"An un-clipped {key} model IS present at {root}. Drop tide_model_layout, "
            f"or set it to 'auto' or 'unclipped', to use it."
        )
    else:
        hint = (
            f"There is no {key} tide model of either layout at {root}. Run "
            f"Download_tide_model.ipynb, then its clipping section."
        )
    shown = ", ".join(missing[:5])
    if len(missing) > 5:
        shown += f" (and {len(missing) - 5} more)"
    raise FileNotFoundError(
        f"No clipped {key} tide model under {root}. tide_model_layout='clipped' asks "
        f"for the legacy layout: region0 through region10 directly under that folder, "
        f"each holding {folders[0]}/ocean_tide[_<YYYYMMDD>] and {folders[0]}/load_tide "
        f"with {_MODEL_LAYOUTS[key]['n_constituents']} .nc files. "
        f"Missing: {shown}. {hint}"
    )


def _find_clipped_regions(
    root: pathlib.Path, model_key: str, model: str
) -> Optional[pathlib.Path]:
    """Find the folder holding clipped regions, at root or one level below it.

    Args:
        root (pathlib.Path): The folder the model was asked for in.
        model_key (str): Canonical model name, a key of _MODEL_LAYOUTS.
        model (str): Tide model name in any accepted spelling.

    Returns:
        pathlib.Path | None: The folder holding region0, or None if there is not one.
    """
    if _model_readable(root / "region0", model):
        return root
    for folder in _MODEL_LAYOUTS[model_key]["folders"]:
        candidate = root / folder
        if _model_readable(candidate / "region0", model):
            return candidate
    # Naming the model folder instead of the model root is the mirror image of clipping
    # in place, and just as easy to do the two differ by exactly this one level. The
    # regions are then one level UP from where we were told to look.
    if root.name in _MODEL_LAYOUTS[model_key]["folders"] and _model_readable(
        root.parent / "region0", model
    ):
        return root.parent
    return None


def _require_clipped_regions(location: Union[str, pathlib.Path], model: str) -> None:
    """Validates the legacy clipped region0 through region10 tide model layout.

    Each region directory is checked with require_tide_groups, the same check
    that is applied to the un-clipped model.

    Args:
        location (str | pathlib.Path): Path to the directory containing the
            region directories.
        model (str): The tide model to check for, in any accepted spelling.

    Returns:
        None

    Raises:
        Exception: If a region directory is missing, or does not contain a
            complete tide model.
    """
    root = pathlib.Path(location)
    for index in range(_N_TIDE_REGIONS):
        region_dir = root / f"region{index}"
        if not region_dir.is_dir():
            raise Exception(
                f"Tide Model was not clipped correctly. Missing the directory "
                f"'{region_dir.name}' for region {index} at {location}. This "
                f"indicates the tide model was not downloaded correctly try again"
            )
        try:
            require_tide_groups(region_dir, model)
        except (FileNotFoundError, ValueError) as exc:
            raise Exception(
                f"Tide Model was not clipped correctly. Region {index} at "
                f"{region_dir} is not usable. {exc} Please download and clip again"
            ) from exc


def get_tide_model_location(
    location: str = "",
    model: str = "fes2022",
    tide_model_layout: Optional[str] = "auto",
) -> str:
    """Validates the existence of a tide model at the specified location and returns the absolute path to the folder.

    Ensures the tide model exists at the given location and validates it contains all the necessary files.

    This function checks if a tide model exists at the given location. If the model exists,
    it returns the absolute path of the location. If the model does not exist, it raises an exception.

    Args:
        location (str, optional): The location to check for the tide model.
                                If empty, defaults to "tide_model" directory in the CoastSeg base directory.
        model (str, optional): The tide model to use. Defaults to 'fes2022'.
                              Available options: 'fes2022', 'fes2014'.
        tide_model_layout (str, optional): Which layout to require: 'auto' (the
            default the un-clipped model, falling back to the clipped regions when
            that is all there is), 'unclipped', or 'clipped'. An explicit choice that
            is not installed raises rather than falling back.

    Returns:
        str: The absolute path of the location if the tide model exists.

    Raises:
        Exception: If the tide model does not exist at the specified location.
        FileNotFoundError: If tide_model_layout names a layout that is not installed.

    Example:
        >>> model_path = get_tide_model_location("/path/to/model", "fes2022")
        >>> print(model_path)
        '/path/to/model'

        >>> # Using default location
        >>> model_path = get_tide_model_location()
        >>> print(model_path)
        '/path/to/coastseg/tide_model'
    """
    location, _ = locate_tide_model(
        location, model=model, tide_model_layout=tide_model_layout
    )
    print(f"Tide model {model} found at: '{location}' and is valid.")
    return location


def locate_tide_model(
    location: str = "",
    model: str = "fes2022",
    tide_model_layout: Optional[str] = "auto",
) -> Tuple[str, str]:
    """The tide model folder to read, and the layout to read it with.

    The single place that decides both. get_tide_model_location and
    resolve_model_layout used to be called in sequence, each probing disk and each
    deciding the layout independently, which left them free to disagree.

    Args:
        location (str, optional): Folder holding the tide model. Empty means the
            default, CoastSeg/tide_model. Keep this the first positional parameter,
            because tests/conftest.py's tide-model guard reads it to decide whether
            a test touches the real model.
        model (str, optional): Tide model name in any accepted spelling.
        tide_model_layout (str, optional): 'auto' (the default), 'unclipped' or
            'clipped'. A named layout that is not installed raises rather than
            falling back to the other one.

    Returns:
        Tuple[str, str]: The absolute model folder, and 'global' or 'regions'.

    Raises:
        Exception: If there is no tide model at that location.
        FileNotFoundError: If tide_model_layout names a layout that is not installed,
            or the chosen layout is incomplete.
    """
    if not location:
        base_dir = os.path.abspath(core_utilities.get_base_dir())
        location = os.path.join(base_dir, "tide_model")
    location = os.path.abspath(location)

    logger.info(f"Checking if tide model exists at {location}")
    if not os.path.isdir(location):
        raise Exception(
            f"Tide model not found at: '{location}' "
            f"(layout requested: {normalize_tide_model_layout(tide_model_layout)}). "
            "Ensure the model is downloaded to this location, or pass "
            "tide_model_location to point at the folder holding it."
        )

    layout = resolve_model_layout(location, model, tide_model_layout=tide_model_layout)
    # An explicitly requested layout that is not installed has to say so, naming the
    # folders it looked for, rather than quietly validating against the other one.
    require_model_layout(location, model, tide_model_layout)

    if layout == "global":
        logger.info("found an un-clipped %s model at %s", model, location)
        require_tide_groups(location, model)
    else:
        # LEGACY: the clipped region0..region10 layout.
        _require_clipped_regions(location, model)
    return location, layout


def validate_tide_model_exists(
    location: str, model: str = "fes2022", tide_model_layout: Optional[str] = "auto"
) -> bool:
    """Whether a usable tide model of the requested layout is installed at location.

    A thin bool-returning wrapper over locate_tide_model, which does the work.
    Only a missing directory comes back as False. An incomplete model, or a layout
    that was asked for and is not there, raises with the detail of what was missing.

    Args:
        location (str): The path to the directory to validate.
        model (str, optional): The tide model name to check. Defaults to 'fes2022'.
                              Available options: 'fes2022', 'fes2014'.
        tide_model_layout (str, optional): Which layout to require: 'auto' (the
            default), 'unclipped' or 'clipped'.

    Returns:
        bool: True when the model is usable. False only when location is not a
        directory; every other failure raises.

    Raises:
        Exception: If the tide model structure is invalid, with specific error messages
                  indicating what components are missing.
        FileNotFoundError: If tide_model_layout names a layout that is not installed.

    Example:
        >>> is_valid = validate_tide_model_exists("/path/to/tide_model", "fes2022")
        >>> print(is_valid)
        True

        >>> # Invalid structure
        >>> is_valid = validate_tide_model_exists("/invalid/path", "fes2022")
        >>> print(is_valid)
        False
    """
    location = os.path.abspath(location)
    logger.info(f"Tide model absolute path {location}")
    # A missing folder is the one failure reported as False rather than raised. Every
    # other way of being wrong, an incomplete model or a layout that was asked for
    # and is not installed, raises from locate_tide_model, naming what was missing.
    if not os.path.isdir(location):
        return False

    locate_tide_model(location, model=model, tide_model_layout=tide_model_layout)
    return True


def normalize_model_name(model: str) -> str:
    """Map any spelling of a supported tide model onto its canonical name.

    Accepts the folder spellings ('fes2022b'), the lowercase forms used in the
    settings files, and hyphenated/underscored variants.

    Args:
        model (str): The model name in any of the accepted spellings.

    Returns:
        str: 'FES2022' or 'FES2014'.

    Raises:
        ValueError: If the model is not one CoastSeg supports.

    Example:
        >>> normalize_model_name("fes2022b")
        'FES2022'
    """
    key = str(model).strip().lower().replace("-", "").replace("_", "")
    if key.startswith("fes2022"):
        return "FES2022"
    if key.startswith("fes2014"):
        return "FES2014"
    raise ValueError(
        f"Unsupported tide model {model!r}; expected one of {sorted(_MODEL_LAYOUTS)}"
    )


def rank_release_dirs(names: Iterable[str], group: str) -> List[str]:
    """A tide group's release folders, oldest first, undated lowest of all.

    Directory names are matched against the release pattern for group. Undated
    directories are ranked before dated releases, so the final item represents
    the most recent available release.

    Args:
        names (Iterable[str]): Candidate folder names, e.g. a directory listing.
        group (str): Tide group to keep, e.g. 'ocean_tide'.

    Returns:
        List[str]: Matching names in preference order, so [-1] is the newest.

    Example:
        >>> rank_release_dirs(
        ...     ["ocean_tide", "ocean_tide_20241025", "ocean_tide_extrapolated"],
        ...     "ocean_tide",
        ... )
        ['ocean_tide', 'ocean_tide_20241025']
    """
    # e.g. ocean_tide or ocean_tide_20241025, anchored so it does not also match
    # ocean_tide_extrapolated or ocean_tide_non_structured.
    pattern = re.compile(rf"^{re.escape(group)}(?:_(\d{{8}}))?$")
    dated: Dict[str, str] = {}
    for name in names:
        match = pattern.match(str(name).strip())
        if match:
            # '' for the undated folder sorts below every 8-digit date
            dated[str(name)] = match.group(1) or ""
    return sorted(dated, key=lambda name: (dated[name], name))


def _find_group_dir(model_dir: pathlib.Path, group: str = "ocean_tide") -> pathlib.Path:
    """Find the newest valid directory for a tide group.

    Prefers dated release directories over the undated directory and skips directories
    that contain no constituent files. The reason being AVISO ships each group either
    under its plain name or under a dated re-release (ocean_tide_20241025), and publishes
    each new release as another dated folder beside the old ones.


    Args:
        model_dir (pathlib.Path): Tide model directory, such as e.g. <region>/fes2022b.
        group (str): Tide group name, such as 'ocean_tide' or 'load_tide'.

    Returns:
        pathlib.Path: The folder holding the constituent files.

    Raises:
        FileNotFoundError: If no such folder holding .nc files exists.
    """

    def _holds_files(path: pathlib.Path) -> bool:
        """Return whether a directory contains constituent files."""
        return path.is_dir() and (any(path.glob("*.nc")) or any(path.glob("*.nc.gz")))

    if not model_dir.is_dir():
        raise FileNotFoundError(f"No '{group}' folder: {model_dir} is not a directory")

    candidates = {p.name: p for p in model_dir.iterdir() if p.is_dir()}
    # newest first, skipping any release whose files have not landed
    for name in reversed(rank_release_dirs(candidates, group)):
        if _holds_files(candidates[name]):
            return candidates[name]

    raise FileNotFoundError(
        f"No '{group}' (or '{group}_<date>') folder holding .nc files under {model_dir}"
    )


def build_model_definition(
    directory: Union[str, pathlib.Path], model: str, group: str = "ocean_tide"
) -> dict:
    """Build a pyTMD model definition from a CoastSeg tide directory.

    Uses the constituent files present on disk so the same code supports
    FES2014, FES2022, regional clips, and global downloads.


    Args:
        directory (str | pathlib.Path): The region folder, e.g. tide_model/region3.
        model (str): Tide model name in any accepted spelling.
        group (str): 'ocean_tide' (default) or 'load_tide'.

    Returns:
        dict: A model definition for pyTMD.io.model(directory).from_dict(...),
            with model_file as POSIX paths relative to directory, plus a
            compressed flag that indicates whether the files are gzipped.

    Raises:
        FileNotFoundError: If the model folder or its constituent files are missing.

    Example:
        >>> d = build_model_definition("tide_model/region3", "FES2022")
        >>> d["z"]["model_file"][0]
        'fes2022b/ocean_tide/2n2_fes2022.nc'
    """
    directory = pathlib.Path(directory).expanduser().resolve()
    key = normalize_model_name(model)
    spec = _MODEL_LAYOUTS[key]

    # The model folder, e.g. fes2022b, under whichever of its known spellings the
    # install used.
    model_dir = next(
        (
            directory / folder
            for folder in spec["folders"]
            if (directory / folder).is_dir()
        ),
        None,
    )
    if model_dir is None:
        raise FileNotFoundError(
            f"No {key} folder under {directory}. Expected one of {spec['folders']}."
        )

    group_dir = _find_group_dir(model_dir, group)

    files = sorted(p for p in group_dir.glob("*.nc") if p.is_file())
    compressed = False
    if not files:
        files = sorted(p for p in group_dir.glob("*.nc.gz") if p.is_file())
        compressed = bool(files)
    if not files:
        raise FileNotFoundError(f"No constituent files in {group_dir}")
    if len(files) != spec["n_constituents"]:
        logger.warning(
            "%s %s holds %d constituent files, expected %d",
            key,
            group_dir,
            len(files),
            spec["n_constituents"],
        )

    # pathlib.Path.glob rejects backslashes, and pyTMD resolves every model_file
    # entry through it, so relative POSIX paths are required.
    relative = [p.relative_to(directory).as_posix() for p in files]
    for rel in relative:
        if any(ch in rel for ch in "*?[]"):
            raise ValueError(f"Tide model path contains glob metacharacters: {rel}")

    return {
        "name": key,
        "format": spec["format"],
        "version": spec["version"],
        "compressed": compressed,
        "z": {
            "model_file": relative,
            "units": spec["units"],
            "variable": spec["variable"],
        },
    }


def resolve_tide_model(
    directory: Union[str, pathlib.Path], model: str, group: str = "ocean_tide"
) -> "pyTMD.io.model":
    """Return a validated pyTMD.io.model for a CoastSeg tide folder.

    Args:
        directory (str | pathlib.Path): The region folder, e.g. tide_model/region3.
        model (str): Tide model name in any accepted spelling.
        group (str): 'ocean_tide' (default) or 'load_tide'.

    Returns:
        pyTMD.io.model: Model with all constituent file paths resolved.

    Raises:
        FileNotFoundError: If the group is incomplete or any constituent file cannot be resolved.

    """
    directory = pathlib.Path(directory).expanduser().resolve()
    # get model definition which specifies how the tide model is structured
    definition = build_model_definition(directory, model, group=group)
    _check_group_is_complete(directory, normalize_model_name(model), group, definition)
    compressed = definition.pop("compressed", False)
    expected = len(definition["z"]["model_file"])

    m = pyTMD.io.model(directory, compressed=compressed, verify=False)
    m = m.from_dict(definition)

    # get the number of constituent files
    resolved = list(m["z"].model_file)
    if len(resolved) != expected:
        raise FileNotFoundError(
            f"pyTMD resolved {len(resolved)} of {expected} constituent files under "
            f"{directory}. The tide model is incomplete try downloading and "
            f"clipping it again."
        )
    return m


def clear_tide_model_cache() -> None:
    """Close and drop every cached tide dataset.

    Call this from tests, and before deleting or re-downloading a tide model on
    Windows, where open netCDF handles keep the files locked.
    """
    with _DATASET_CACHE_LOCK:
        while _DATASET_CACHE:
            _, (_, ds) = _DATASET_CACHE.popitem()
            try:
                ds.close()
            except Exception:
                logger.debug("failed to close a cached tide dataset", exc_info=True)


atexit.register(clear_tide_model_cache)


def _open_tide_dataset(
    directory: Union[str, pathlib.Path], model: str, group: str = "ocean_tide"
) -> Tuple[object, "xr.Dataset"]:  # noqa F821
    """Return (model, dataset) for a region directory and cache it.

    Opens the dataset lazily when possible and reuses it for subsequent calls with
    the same directory, model, and group.

    Args:
        directory (str | pathlib.Path): Folder holding the tide model.
        model (str): Tide model name in any accepted spelling.
        group (str): Constituent group, 'ocean_tide' or 'load_tide'.

    Returns:
        Tuple[pyTMD.io.model, xr.Dataset]: The resolved model and its lazily opened
            dataset

    Note:
        The cache itself is thread-safe, but the cached datasets are not safe for
        concurrent reads.
    """
    key = (
        str(pathlib.Path(directory).expanduser().resolve()),
        normalize_model_name(model),
        group,
    )
    with _DATASET_CACHE_LOCK:
        hit = _DATASET_CACHE.get(key)
        if hit is not None:
            _DATASET_CACHE.move_to_end(key)
            return hit

    # Opened outside the lock: reading 34 file headers should not block other ROIs.
    m = resolve_tide_model(directory, model, group=group)
    try:
        ds = m.open_dataset(group="z", chunks={})
    except (ImportError, ValueError):
        logger.warning(
            "lazy tide model read failed; falling back to an eager read. Large "
            "regions may need several GB of RAM.",
            exc_info=True,
        )
        ds = m.open_dataset(group="z")

    with _DATASET_CACHE_LOCK:
        existing = _DATASET_CACHE.get(key)
        if existing is not None:
            try:
                ds.close()
            except Exception:
                pass
            _DATASET_CACHE.move_to_end(key)
            return existing
        _DATASET_CACHE[key] = (m, ds)
        while len(_DATASET_CACHE) > TIDE_DATASET_CACHE_SIZE:
            _, (_, evicted) = _DATASET_CACHE.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                logger.debug("failed to close an evicted tide dataset", exc_info=True)
        return _DATASET_CACHE[key]


def read_and_filter_geojson(
    file_path: str,
    columns_to_keep: Tuple[str, ...] = ("id", "type", "geometry"),
    feature_type: str = "transect",
) -> gpd.GeoDataFrame:
    """Read and filter a GeoJSON file based on specified columns and feature type.

    Args:
        file_path (str): Path to the GeoJSON file.
        columns_to_keep (Tuple[str, ...], optional): Column names to be retained in the
            resulting GeoDataFrame. Defaults to ("id", "type", "geometry").
        feature_type (str, optional): Type of feature to be retained in the resulting
            GeoDataFrame. Defaults to "transect".

    Returns:
        gpd.GeoDataFrame: A filtered GeoDataFrame.
    """
    # Read the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(file_path)
    # Drop all other columns in place
    gdf.drop(
        columns=[col for col in gdf.columns if col not in columns_to_keep], inplace=True
    )
    # Filter the features with "type" equal to the specified feature_type
    filtered_gdf = gdf[gdf["type"] == feature_type]

    return filtered_gdf


def timeseries_read_csv(file_path: str, is_merged: bool = False) -> pd.DataFrame:
    """Reads the timeseries from a CSV file.

    It converts the dates column to datetime in UTC.
    It drops the columns 'x', 'y', and 'Unnamed: 0' if they exist.

    Args:
        file_path (str): Path to the CSV file.
        is_merged (bool, optional): Indicates if the timeseries is merged. Defaults to False.

    Returns:
        pd.DataFrame: Processed data.
    """
    df = pd.read_csv(file_path)
    if "dates" not in df.columns:
        raise ValueError(
            f"{file_path} has no 'dates' column, so it is not a CoastSeg time "
            f"series. Columns found: {list(df.columns)}"
        )
    df["dates"] = pd.to_datetime(df["dates"], utc=True)
    # only for the non merged timeseries do we drop the x, y and Unnamed: 0 columns
    if is_merged is False:
        for column in ["x", "y", "Unnamed: 0"]:
            if column in df.columns:
                df.drop(columns=column, inplace=True)
    return df
