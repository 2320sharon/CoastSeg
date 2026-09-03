"""CoastSeg Tide and Slope File Reading

Reads user supplied tide and slope CSV files and matches their values onto
shoreline observations.

The supported file layouts are listed in _SEASONAL_FORMATS and _DATED_FORMATS.
Adding a layout requires one entry in the appropriate table and the matching
function it names. The layouts are documented at
https://satelliteshorelines.github.io/CoastSeg/tide-file-format/
"""

# Standard library imports
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Mean Earth radius in kilometres, for the great-circle distance below.
_EARTH_RADIUS_KM = 6371.0088
# Max distance btw two reference locations to count as tied for nearest to tide prediction
_DISTANCE_TIE_RTOL = 1e-4
# Abs floor(km), a transect sitting exactly on a gauge still groups any other gauge at the same spot rather than dividing by a zero baseline.
_DISTANCE_TIE_ATOL_KM = 1e-6
# Private merge key. The caller's own transect_id column has to survive the merge,
# so the string-cast copy the merge keys on travels under a name no file can use.
_TRANSECT_KEY = "_transect_key"


def convert_col_to_ISO_8601(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts a DataFrame column to ISO 8601 format and timezone-aware datetime objects.

    Args:
        df (pd.DataFrame): The input DataFrame containing the column to convert.
        col_name (str): Name of the column to convert to ISO 8601 format.

    Returns:
        pd.DataFrame: DataFrame with the specified column converted to timezone-aware datetime objects in UTC.

    """
    if col_name not in df.columns:
        return df
    # utc=True covers both halves. A naive stamp is read as UTC, an offset stamp is
    # converted to it.
    df[col_name] = pd.to_datetime(df[col_name], format="ISO8601", utc=True)
    return df


def compute_distance_lonlat(
    lon1: Union[float, np.ndarray],
    lat1: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
) -> np.ndarray:
    """Compute great-circle distance between longitude/latitude points.

    Uses the haversine formula and supports scalar or broadcastable array inputs

    Args:
        lon1 (float | np.ndarray): Longitude of the first point(s), in degrees.
        lat1 (float | np.ndarray): Latitude of the first point(s), in degrees.
        lon2 (float | np.ndarray): Longitude of the second point(s), in degrees.
        lat2 (float | np.ndarray): Latitude of the second point(s), in degrees.

    Returns:
        np.ndarray | float: Distance in kilometres.

    Example:
        >>> round(float(compute_distance_lonlat(179.9, 0.0, -179.9, 0.0)), 1)
        22.2
    """
    lon1, lat1, lon2, lat2 = map(
        lambda v: np.radians(np.asarray(v, dtype=float)), (lon1, lat1, lon2, lat2)
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # haversine: numerically stable for the small separations we care about
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _initialize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Returns a copy of DataFrame with the specified column initialized with NaN values if it was absent.

    Args:
        df (pd.DataFrame): DataFrame to base the copy on.
        column_name (str): Name of the column to initialize.

    Returns:
        pd.DataFrame: A copy of the DataFrame containing the specified column, by column_name
    """
    df = df.copy()
    if column_name not in df.columns:
        df[column_name] = np.nan
    return df


def _nearest_by_date(
    timeseries: pd.DataFrame,
    df: pd.DataFrame,
    column_name: str,
    by: Optional[str] = None,
) -> pd.Series:
    """Match each timeseries row to the temporally nearest reference row.

    If by is provided, rows are first matched on that column. When two
    reference rows are equally close in time, the earlier one is selected.

    Args:
        timeseries (pd.DataFrame): Rows to match, with a 'dates' column.
        df (pd.DataFrame): Reference rows, with 'dates' and column_name.
        column_name (str): Name of the reference column to return.
        by (str, optional): Extra key to match exactly first, e.g. 'transect_id'.
            Rows whose key is absent from df come back as NaN.

    Returns:
        pd.Series: Matched values aligned to the index of timeseries.

    Example:
        >>> timeseries = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(["2024-01-02", "2024-01-08"]),
        ...         "transect_id": ["A", "B"],
        ...     }
        ... )
        >>> df = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(
        ...             ["2024-01-01", "2024-01-10", "2024-01-10", "2024-01-01"]
        ...         ),
        ...         "transect_id": ["A", "A", "B", "B"],
        ...         "tide": [1.2, 1.8, 2.5, 2.1],
        ...     }
        ... )
        >>> _nearest_by_date(
        ...     timeseries,
        ...     df,
        ...     column_name="tide",
        ...     by="transect_id",
        ... ).tolist()
        [1.2, 2.5]
    """
    # keep only the dates, column name and an optional by column
    keep = ["dates", column_name] + ([by] if by else [])
    # Sort both the reference and timeseries by date so we can safely merge them
    right = df[keep].dropna(subset=["dates"]).sort_values("dates", kind="mergesort")
    # drop any duplicate entries in the reference and only keep the first one. Eg if the same transect_id and dates occur multiple times keep the first one
    right = right.drop_duplicates(subset=["dates"] + ([by] if by else []), keep="first")
    # Use a stable sort, mergesort to perserve the original order of rows with equal dates eg. if two dates are the same keep them in the same order they were
    left = timeseries[["dates"] + ([by] if by else [])].sort_values(
        "dates", kind="mergesort"
    )
    # Only keep valid dates so merge_asof will work and leave index intact so we can add NaNs back later (no NANs allowed in left)
    left = left[left["dates"].notna()]

    if right.empty or left.empty:
        return pd.Series(np.nan, index=timeseries.index, dtype="float64")
    # Match based on the closest date and by transect_id (aka variable `by`)
    merged = pd.merge_asof(
        left,
        right,
        on="dates",
        by=by,  # match transect_ids first before comparing dates
        direction="nearest",  # check for nearest time backwards and forwards
        suffixes=("", "_ref"),
    )
    # Restore original index order that left index had so that NaN dates can be restored
    merged.index = left.index
    # Add back the dropped NaN dates that are still in the left index ( missing indexes get NaN)
    return merged[column_name].reindex(timeseries.index)


def _match_via_season(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str, keys: List[str]
) -> pd.DataFrame:
    """Matches values based on the calendar month, and optionally the transect ID.

    The reference data is reduced to one row per key before merging, so that the
    merge cannot duplicate observations. The index of the timeseries is restored
    afterwards because merging renumbers the rows.

    Args:
        timeseries (pd.DataFrame): DataFrame containing shoreline data with a
            'dates' column.
        df (pd.DataFrame): Reference DataFrame with a 'month' column and the
            specified column.
        column_name (str): Name of the column to match and add to the timeseries.
        keys (List[str]): Columns to merge on, either ['month'] or
            ['month', 'transect_id'].

    Returns:
        pd.DataFrame: A copy of timeseries with the specified column added. Rows
            that the reference does not cover carry NaN; nothing is substituted
            for them.
    """
    timeseries = timeseries.copy()
    # if the column already exists drop it else the merge appends a duplicate (eg slope_y)
    if column_name in timeseries.columns:
        timeseries = timeseries.drop(columns=[column_name], errors="ignore")

    by_transect = "transect_id" in keys
    if by_transect:
        df = df.assign(transect_id=df["transect_id"].astype(str))

    # One reference row per key, or the merge below multiplies the time series.
    # Eg. two January rows for the same transect become their mean.
    duplicated = int(df.duplicated(subset=keys).sum())
    if duplicated:
        logger.warning(
            "%d row(s) in the %s file share the same %s; averaging them so the merge "
            "does not duplicate observations in the corrected time series",
            duplicated,
            column_name,
            " and ".join(keys),
        )
        reference = df.groupby(keys, as_index=False, sort=False)[[column_name]].mean()
    else:
        reference = df

    # Both sides of the merge key have to be strings, and the caller's own
    # transect_id column must survive, so the key travels under its own name.
    merge_on = list(keys)
    if by_transect:
        reference = reference.rename(columns={"transect_id": _TRANSECT_KEY})
        merge_on = [_TRANSECT_KEY if k == "transect_id" else k for k in keys]
        timeseries[_TRANSECT_KEY] = timeseries["transect_id"].astype(str)

    index = timeseries.index
    timeseries["month"] = timeseries["dates"].dt.month
    timeseries = timeseries.merge(reference, on=merge_on, how="left")
    # merge renumbers the rows; the reference holds one row per key, so the result
    # is the caller's rows in the caller's order and the index can be put back.
    timeseries.index = index
    return timeseries.drop(columns=["month"] + ([_TRANSECT_KEY] if by_transect else []))


def match_via_month(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str = "slope"
) -> pd.DataFrame:
    """Matches timeseries data with reference data based on month and adds the specified column.

    Args:
        timeseries (pd.DataFrame): DataFrame containing timeseries data with 'dates' column.
        df (pd.DataFrame): Reference DataFrame with 'month' column and specified column for matching.
        column_name (str, optional): Name of the column to match and add. Defaults to 'slope'.

    Returns:
        pd.DataFrame: Timeseries DataFrame with the specified column added, temporary 'month' column removed.

    Example:
        >>> timeseries_df = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(["2021-01-15", "2021-06-15"]),
        ...         "transect_id": ["1", "2"],
        ...     }
        ... )
        >>> slope_df = pd.DataFrame({"month": [1, 6], "slope": [0.1, 0.2]})
        >>> result = match_via_month(timeseries_df, slope_df, "slope")
        >>> print(result)
            dates transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           2    0.2
    """
    return _match_via_season(timeseries, df, column_name, ["month"])


def match_via_id_and_month(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """Matches values based on transect_id and closest month.

    Args:
        timeseries (pd.DataFrame): DataFrame containing shoreline data with 'dates' and 'transect_id' columns.
        df (pd.DataFrame): Reference DataFrame with 'transect_id', 'month', and specified column.
        column_name (str): Name of column to match and add to the timeseries.

    Returns:
        pd.DataFrame: A copy of timeseries with matched values. Rows the reference
                     does not cover carry NaN; nothing is substituted for them.
                     The frame passed in is left unmodified.

    Example:
        >>> timeseries_df = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(["2021-01-15", "2021-06-15"]),
        ...         "transect_id": ["1", "1"],
        ...     }
        ... )
        >>> slope_df = pd.DataFrame(
        ...     {"transect_id": ["1", "1"], "month": [1, 6], "slope": [0.1, 0.2]}
        ... )
        >>> result = match_via_id_and_month(timeseries_df, slope_df, "slope")
        >>> print(result)
          dates        transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           1    0.2
    """
    return _match_via_season(timeseries, df, column_name, ["month", "transect_id"])


def match_via_date(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """Matches values based on closest date.

    Args:
        timeseries (pd.DataFrame): DataFrame containing shoreline data with 'dates' column.
        df (pd.DataFrame): Reference DataFrame with 'dates' and specified column.
        column_name (str): Name of column to match and add to the timeseries.

    Returns:
        pd.DataFrame: A copy of timeseries with matched values added. The frame
            passed in is left unmodified.

    Example:
        >>> timeseries_df = pd.DataFrame(
        ...     {"dates": pd.to_datetime(["2021-01-15", "2021-06-15"])}
        ... )
        >>> reference_df = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(["2021-01-10", "2021-06-20"]),
        ...         "tide": [1.2, 1.8],
        ...     }
        ... )
        >>> result = match_via_date(timeseries_df, reference_df, "tide")
        >>> print(result)
            dates  tide
        0 2021-01-15   1.2
        1 2021-06-15   1.8
    """
    timeseries = _initialize_column(timeseries, column_name)
    timeseries[column_name] = _nearest_by_date(timeseries, df, column_name)
    return timeseries


def match_via_id_and_date(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """Matches values based on transect_id and closest date.

    Performs temporal matching for each transect, finding the closest date
    in the reference DataFrame for each shoreline observation.

    Args:
        timeseries (pd.DataFrame): DataFrame containing shoreline data with 'dates' and 'transect_id' columns.
        df (pd.DataFrame): Reference DataFrame with 'dates', 'transect_id', and specified column.
        column_name (str): Name of column to match and add to the timeseries.

    Returns:
        pd.DataFrame: A copy of timeseries with matched values. Rows the reference
                     does not cover carry NaN; nothing is substituted for them.
                     The frame passed in is left unmodified.

    Example:
        >>> timeseries_df = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(["2021-01-15", "2021-06-15"]),
        ...         "transect_id": ["1", "1"],
        ...     }
        ... )
        >>> reference_df = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(["2021-01-10", "2021-06-20"]),
        ...         "transect_id": ["1", "1"],
        ...         "slope": [0.1, 0.2],
        ...     }
        ... )
        >>> result = match_via_id_and_date(timeseries_df, reference_df, "slope")
        >>> print(result)
            dates transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           1    0.2
    """
    timeseries = _initialize_column(timeseries, column_name)
    reference = df.copy()
    reference["transect_id"] = reference["transect_id"].astype(str)
    matched = _nearest_by_date(
        timeseries.assign(transect_id=timeseries["transect_id"].astype(str)),
        reference,
        column_name,
        by="transect_id",
    )
    timeseries[column_name] = matched
    return timeseries


def match_via_points_and_date(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """Matches measurements to transects based on closest spatial and temporal proximity.

    Finds spatially closest points in the reference DataFrame for each transect,
    then temporally matches each shoreline observation to the closest measurement.

    Args:
        timeseries (pd.DataFrame): DataFrame with columns: 'transect_id', 'x', 'y', 'dates'.
        df (pd.DataFrame): Reference DataFrame with columns: 'latitude', 'longitude', 'dates', and specified column.
        column_name (str): Name of the column in df containing the values to match.

    Returns:
        pd.DataFrame: A copy of timeseries with the specified column added, containing
            matched values. The frame passed in is left unmodified.

    Example:
        >>> timeseries_df = pd.DataFrame(
        ...     {
        ...         "transect_id": ["1", "1"],
        ...         "x": [-120.5, -120.5],
        ...         "y": [35.2, 35.2],
        ...         "dates": pd.to_datetime(["2021-01-15", "2021-06-15"]),
        ...     }
        ... )
        >>> reference_df = pd.DataFrame(
        ...     {
        ...         "latitude": [35.19, 35.21],
        ...         "longitude": [-120.51, -120.49],
        ...         "dates": pd.to_datetime(["2021-01-10", "2021-06-20"]),
        ...         "tide": [1.2, 1.8],
        ...     }
        ... )
        >>> result = match_via_points_and_date(timeseries_df, reference_df, "tide")
        >>> print(result)
        transect_id      x     y      dates  tide
        0           1  -120.5  35.2 2021-01-15   1.2
        1           1  -120.5  35.2 2021-06-15   1.8
    """
    timeseries = _initialize_column(timeseries, column_name)
    if timeseries.empty or df.empty:
        return timeseries

    # One representative coordinate per transect, in the caller's row order.
    transects = timeseries.groupby("transect_id", sort=False)[["x", "y"]].first()

    # Assign each unique reference location an integer ID.
    locations = df[["longitude", "latitude"]].drop_duplicates().reset_index(drop=True)
    location_of_row = df.merge(
        locations.reset_index(names="_loc"), on=["longitude", "latitude"], how="left"
    )["_loc"].to_numpy()

    # Compute  great-circle distances from every transect to every reference location. Eg (transects x locations)
    distances = compute_distance_lonlat(
        transects["x"].to_numpy()[:, None],
        transects["y"].to_numpy()[:, None],
        locations["longitude"].to_numpy()[None, :],
        locations["latitude"].to_numpy()[None, :],
    )

    # # Keep all locations tied for the minimum distance. Ties are decided with relative tolerance; see _DISTANCE_TIE_RTOL
    closest = distances.min(axis=1, keepdims=True)
    nearest = distances <= closest * (1.0 + _DISTANCE_TIE_RTOL) + _DISTANCE_TIE_ATOL_KM

    # Group transects that share the same nearest reference location because they are interchangeable for the
    # temporal match, so group them and expand the reference once per group.
    # Expanding once per transect instead would repeat the reference rows n_transects
    # times. 300 transects against 300k readings is 90M rows.
    group_of_transect: Dict[str, int] = {}
    group_locations: Dict[tuple, int] = {}
    for row, transect_id in enumerate(transects.index):
        key = tuple(np.flatnonzero(nearest[row]))
        group_of_transect[transect_id] = group_locations.setdefault(
            key, len(group_locations)
        )
    # Build one temporal reference table per location group.
    tagged = df.assign(_loc=location_of_row)[["_loc", "dates", column_name]]
    reference = pd.concat(
        [
            tagged[tagged["_loc"].isin(key)].drop(columns="_loc").assign(_grp=group_id)
            for key, group_id in group_locations.items()
        ],
        ignore_index=True,
    )

    # Match each observation to the nearest date within its location group.
    left = timeseries[["dates"]].assign(
        _grp=timeseries["transect_id"].map(group_of_transect)
    )
    timeseries[column_name] = _nearest_by_date(left, reference, column_name, by="_grp")
    return timeseries


def _find_dates_column(df: pd.DataFrame) -> Optional[str]:
    """The column of a wide (dates x transect_id) frame that holds the dates.

    Args:
        df (pd.DataFrame): Frame to inspect.

    Returns:
        str | None: the 'dates' column in any capitalisation, otherwise the leftmost
            unnamed index column in any capitalisation, otherwise None.

    Note:
        The capitalisation is matched loosely because the frame reaching here has not
        been case folded: in a matrix the headers are transect ids, and folding their
        case would stop them matching the time series.
    """
    _UNNAMED_INDEX_COLUMN = re.compile(r"^unnamed:\s*\d+$", re.IGNORECASE)
    for column in df.columns:
        if isinstance(column, str) and column.strip().lower() == "dates":
            return column
    for column in df.columns:
        if isinstance(column, str) and _UNNAMED_INDEX_COLUMN.match(column.strip()):
            return column
    return None


def melt_df(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Transforms a DataFrame by melting it, converting columns into rows.

    Handles two formats: the dates in a column ('dates', or the unnamed index column
    pandas writes as 'Unnamed: 0' in any capitalisation), or the dates on the row
    index. Converts the 'dates' column to datetime format and melts the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to be transformed.
        column_name (str): Name for the values column in the melted DataFrame.

    Returns:
        pd.DataFrame: Melted DataFrame with 'dates', 'transect_id', and specified column.

    Raises:
        ValueError: If no column holds the dates and the index is positional, so
            there is nothing to melt against. Falling back to the index in that case
            produced epoch timestamps rather than the file's real dates.

    Example:
        >>> df = pd.DataFrame(
        ...     {
        ...         "Unnamed: 0": ["2021-01-01", "2021-01-02"],
        ...         "1": [1.2, 1.3],
        ...         "2": [1.3, 1.4],
        ...     }
        ... )
        >>> result = melt_df(df, "tide")
        >>> print(result)
               dates transect_id  tide
        0 2021-01-01           1   1.2
        1 2021-01-02           1   1.3
        2 2021-01-01           2   1.3
        3 2021-01-02           2   1.4
    """

    dates_column = _find_dates_column(df)
    # if a dates column exists rename to dates
    if dates_column is not None:
        df = df.rename(columns={dates_column: "dates"})
    elif not isinstance(df.index, pd.RangeIndex):
        # Turn the index column which contained dates into a column called dates and resetss the index
        df = df.reset_index(names="dates")
    else:
        # this means neither the index nor any column carried any dates. This means the df is not a dates x transect_id matrix.
        raise ValueError(
            "Cannot read this file as a dates x transect_id matrix: no 'dates' "
            "column, no unnamed index column, and the row index is positional. "
            f"Columns found: {list(df.columns)}"
        )
    df["dates"] = pd.to_datetime(df["dates"])
    df = pd.melt(df, id_vars=["dates"], var_name="transect_id", value_name=column_name)
    return df


def clean_dataframe(
    df: pd.DataFrame,
    keep_columns: Union[List[str], None] = None,
    convert_to_lower: bool = True,
    remove_s: bool = True,
) -> pd.DataFrame:
    """Cleans DataFrame by transforming column names and filtering columns.

    Args:
        df (pd.DataFrame): DataFrame to be cleaned.
        keep_columns (Union[List[str], None], optional): Column names to retain.
            If None, keeps all columns after cleaning. Defaults to None.
        convert_to_lower (bool, optional): Whether to convert column names to lowercase. Defaults to True.
        remove_s (bool, optional): Accept plural spellings of the wanted columns, so
            'slopes' is read as 'slope'. Only strips the 's' when doing so produces a
            name in keep_columns, and so does nothing when keep_columns is
            None. It used to strip a trailing 's' from *every* column, which turned
            an unrelated 'address' into 'addres'. Defaults to True.

    Returns:
        pd.DataFrame: Cleaned DataFrame with transformed column names and optionally filtered columns.

    Example:
        >>> df = pd.DataFrame(
        ...     {
        ...         "TransectIds": [1, 2, 3],
        ...         "Slopes": [0.1, 0.2, 0.3],
        ...         "ExtraColumn": ["a", "b", "c"],
        ...     }
        ... )
        >>> result = clean_dataframe(df, keep_columns=["transectid", "slope"])
        >>> print(result.columns.tolist())
        ['transectid', 'slope']
    """
    # Renaming and dropping used to happen on the caller's frame, so a caller that
    # still held it found its columns lowercased and its trailing 's' stripped.
    df = df.copy()
    if convert_to_lower:
        df.columns = df.columns.str.lower()

    if remove_s and keep_columns is not None:
        wanted = set(keep_columns)
        df.columns = [
            (
                col[:-1]
                if (
                    isinstance(col, str)
                    and col not in wanted
                    and col.endswith("s")
                    and col[:-1] in wanted
                )
                else col
            )
            for col in df.columns
        ]
    if keep_columns is None:
        return df
    cols_to_drop = [col for col in df.columns if col not in keep_columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def _looks_like_matrix(raw: pd.DataFrame, timeseries: pd.DataFrame) -> bool:
    """True when a frame's headers are transect ids, i.e. it is a wide matrix.

    A matrix and a file that holds an unrecognised value column look alike:
    dates in one column, numbers in the others. A real matric has headers
    that name transects the time series.

    Args:
        raw (pd.DataFrame): The file as read, with its headers un-case-folded.
        timeseries (pd.DataFrame): The observations the file will be matched against.

    Returns:
        bool: True when at least one header names a transect in the time series, or
            when the time series carries no transect ids to compare against.
    """
    if "transect_id" not in timeseries.columns:
        # nothing to compare against, so fall back to the shape alone
        return True
    dates_column = _find_dates_column(raw)
    candidates = {str(c) for c in raw.columns if c != dates_column}
    known = {str(t) for t in timeseries["transect_id"]}
    return not candidates.isdisjoint(known)


def _unsupported_format_error(column_name: str, detail: str = "") -> ValueError:
    """The documented "this file layout is not supported" error for a value column.

    Args:
        column_name (str): The column being read, 'tide' or 'slope'.
        detail (str, optional): What specifically was wrong with this file, for the
            cases where the layout is recognisable and something else went wrong.
            Inserted before the link to the documentation.

    Returns:
        ValueError: Ready to raise, pointing at the matching documentation page.
    """
    page = "tide-file-format" if column_name == "tide" else "slope-file-format"
    reason = f" {detail}" if detail else ""
    return ValueError(
        f"CSV format not supported.{reason} Must be in one of the following formats "
        f"as listed on the documentation: "
        f"https://satelliteshorelines.github.io/CoastSeg/{page}/"
    )


# The tide/slope file layouts CoastSeg accepts. The first entry whose key columns
# are all present wins, so the more specific layouts come first. Adding a layout
# is one line here plus the matcher it names.
#
# There are two tables because the two families need different preparation: a
# seasonal file is de-pluralised and cut down to its three known columns, while a
# dated file keeps its coordinates and has its stamps parsed to UTC first.
_Matcher = Callable[[pd.DataFrame, pd.DataFrame, str], pd.DataFrame]

#: Layouts carrying a calendar month rather than a date.
_SEASONAL_FORMATS: Tuple[Tuple[Tuple[str, ...], _Matcher], ...] = (
    (("transect_id", "month"), match_via_id_and_month),
    (("month",), match_via_month),
)

#: Layouts carrying one row per reading, matched on the nearest date.
_DATED_FORMATS: Tuple[Tuple[Tuple[str, ...], _Matcher], ...] = (
    (("transect_id", "dates"), match_via_id_and_date),
    (("latitude", "dates"), match_via_points_and_date),
    (("dates",), match_via_date),
)


def _pick_matcher(
    formats: Tuple[Tuple[Tuple[str, ...], _Matcher], ...], columns: Iterable[str]
) -> Optional["_Matcher"]:
    """Returns the matching function for the first layout the columns satisfy.

    Args:
        formats (tuple): Supported layouts, as (required columns, matching
            function) pairs, ordered with the most specific layout first.
        columns (Iterable[str]): The column names present in the file.

    Returns:
        Optional[_Matcher]: The matching function to use, or None if the columns
            do not match any supported layout.
    """
    present = set(columns)
    for keys, matcher in formats:
        if present.issuperset(keys):
            return matcher
    return None


def read_content_csv(
    file: Union[str, Path], timeseries: pd.DataFrame, column_name: str = "tide"
) -> pd.DataFrame:
    """Reads CSV data and merges it with timeseries DataFrame using appropriate matching strategy.

    Supports various CSV formats including seasonal data with monthly information
    and time series data with different matching strategies based on available columns.

    Args:
        file (Union[str, Path]): Path to the CSV file containing data to be added to timeseries.
        timeseries (pd.DataFrame): DataFrame containing shoreline data.
        column_name (str, optional): Name of column to match and add. Defaults to 'tide'.

    Returns:
        pd.DataFrame: DataFrame containing merged data from the CSV file with the timeseries.

    Raises:
        ValueError: If the CSV format is not supported or doesn't match expected column structures.

    Example:
        >>> timeseries_df = pd.DataFrame(
        ...     {
        ...         "dates": pd.to_datetime(["2021-01-15", "2021-06-15"]),
        ...         "transect_id": ["1", "2"],
        ...     }
        ... )
        >>> # CSV file with monthly data
        >>> result = read_content_csv("monthly_data.csv", timeseries_df, "slope")
        >>> print(result)
            dates transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           2    0.2
    """
    timeseries = timeseries.copy()

    raw = pd.read_csv(file)

    # clean the dataframe to ensure a consistent format
    df = clean_dataframe(raw, keep_columns=None, remove_s=False)

    # A seasonal file is recognised by carrying a month rather than a date.
    if any(df.columns.str.contains(r"(?i)month")):
        df = clean_dataframe(df, keep_columns=["transect_id", "month", column_name])
        matcher = (
            _pick_matcher(_SEASONAL_FORMATS, df.columns)
            if column_name in df.columns
            else None
        )
        if matcher is None:
            raise ValueError(
                f'CSV format not supported. If you are using a CSV file with monthly data then the columns should be "month" and "{column_name}" or  "transect_id", "month" and "{column_name}"'
            )
        return matcher(timeseries, df, column_name)

    # A wide matrix carries neither the value column nor transect_id, because its
    # headers ARE the transect ids. Reshape it into the long form the dated
    # matchers expect.
    if column_name not in df.columns and "transect_id" not in df.columns:
        # validate the matrix is valid and not just random ids
        if not _looks_like_matrix(raw, timeseries):
            raise _unsupported_format_error(
                column_name,
                f"{file} has no '{column_name}' column, and none of its other "
                f"columns ({[str(c) for c in raw.columns][:10]}) names a transect "
                "in this ROI's time series, so it is neither the long form nor a "
                "dates x transect_id matrix. If it is meant to be a matrix, check "
                "that its column headers are this ROI's transect ids.",
            )
        try:
            df = melt_df(raw, column_name)
        except ValueError as exc:
            # Not a matrix either, so the file matches no supported layout.
            raise _unsupported_format_error(column_name) from exc

    # Keep columns used by the matching strategies.
    df = clean_dataframe(
        df,
        keep_columns=["transect_id", "dates", column_name, "latitude", "longitude"],
        remove_s=False,
    )

    # Convert the dates column to ISO 8601 format and ensure it is timezone-aware
    df = convert_col_to_ISO_8601(df, "dates")

    # A layout with no value column, or no dates to match on, is not supported.
    matcher = (
        _pick_matcher(_DATED_FORMATS, df.columns) if column_name in df.columns else None
    )
    if matcher is None:
        raise _unsupported_format_error(column_name)
    return matcher(timeseries, df, column_name)
