"""CoastSeg Tide Prediction

Predicts tides at points and times using pyTMD 3.

predict_tides converts transects into seaward points and, on the clipped layout,
looks up the regional grid that contains each one. predict_tides_for_df groups
the points by the directory they are read from, _predict_region batches each
group, and model_tides opens the grids once, crops them, interpolates them, and
sums the harmonics.
"""

# Standard library imports
import hashlib
import logging
import pathlib
from collections.abc import Collection
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import timescale
import xarray as xr

# Third-party imports
from tqdm import tqdm

from coastseg import common

# Logger setup
logger = logging.getLogger(__name__)

# Kilometres in a degree of latitude (WGS84 mean). A degree of longitude is this
# times cos(latitude).
_KM_PER_DEGREE = 111.195

# pyTMD 3 only implements 'linear' and 'nearest'. Keep accepting the v2 spellings so existing callers and saved configs do not break.
_METHOD_ALIASES = {
    "bilinear": "linear",
    "spline": "linear",
    "linear": "linear",
    "nearest": "nearest",
}

# Cap on n_points * n_constituents * n_times per prediction block.
_MAX_PREDICT_ELEMENTS = 8_000_000

# Never build a single points x times block larger than this.
_MAX_BATCH_CELLS = 20_000_000

# Only split a batch once the waste exceeds this many cells per extra call the split would create.
_WASTE_CELLS_PER_EXTRA_CALL = 1_700_000


def _tc():
    """Returns the coastseg.tide_correction module.

    The import is deferred to avoid a circular dependency during module
    initialization and to ensure callers access current module-level settings.

    Returns:
        module: The coastseg.tide_correction module.
    """
    from coastseg import tide_correction

    return tide_correction


def _constituents_only(ds: "xr.Dataset") -> "xr.Dataset":
    """Return only harmonic constituent variables.

    The clipped files carry lon_bnds, lat_bnds and crs alongside the
    constituents; those lack the x/y dimensions that interpolation needs.

    Args:
        ds (xr.Dataset): A tide model dataset as opened by pyTMD.

    Returns:
        xr.Dataset: Only the constituent variables, with the dataset attrs kept.
    """
    out = ds[list(ds.tmd.constituents)]
    out.attrs = dict(ds.attrs)
    return out


def _match_lon_convention(ds: "xr.Dataset", lon: np.ndarray) -> np.ndarray:
    """Shift query longitudes into the model grid's longitude convention.

    Grids are either -180..180 or 0..360, so a point at -118.5 has to be asked
    for as 241.5 on an unclipped AVISO download. pyTMD does this too, but
    `_crop_to_points` slices the grid by longitude first, so it has to happen
    here. Shifting twice is harmless: the second shift finds nothing to do.

    Args:
        ds (xr.Dataset): The tide model dataset, used only for its x coordinate.
        lon (np.ndarray): Query longitudes in EPSG:4326.

    Returns:
        np.ndarray: The longitudes in the grid's convention.
    """
    lon = np.asarray(lon, dtype=float)
    x = np.asarray(ds.coords["x"].values, dtype=float)
    if x.size < 2:
        return lon
    dx = x[1] - x[0]
    if (np.nanmin(lon) < 0.0) and (x.max() > (180.0 + dx)):
        return np.where(lon < 0.0, lon + 360.0, lon)
    if (np.nanmax(lon) > 180.0) and (x.min() < (0.0 - dx)):
        return np.where(lon > 180.0, lon - 360.0, lon)
    return lon


def _materialise(ds: "xr.Dataset") -> "xr.Dataset":
    """Load a lazy dataset into memory, preserving its attributes.

    Args:
        ds (xr.Dataset): Dataset that may be dask-backed.

    Returns:
        xr.Dataset: The same data, computed if it was chunked.
    """
    if any(getattr(ds[v].data, "chunks", None) is not None for v in ds.data_vars):
        attrs = dict(ds.attrs)
        ds = ds.compute()
        ds.attrs = attrs
    return ds


def _is_global_grid(x: np.ndarray) -> bool:
    """Return whether an x coordinate spans a global grid.

    An un-clipped AVISO download spans 0..360; a clipped region covers far less.

    Args:
        x (np.ndarray): The grid's x coordinate, in degrees.

    Returns:
        bool: True when the axis spans the full 360 degrees.
    """
    if x.size < 2:
        return False
    dx = abs(x[1] - x[0])
    return bool((x.max() - x.min()) >= (360.0 - 2.0 * dx))


def _seam_anchor(lon: np.ndarray, origin: float = 0.0) -> Optional[float]:
    """Where to cut a global grid so the requested points come out contiguous.

    A 0..360 grid has a branch cut at Greenwich, so an ROI straddling it looks
    like it spans the planet (0.05 and 359.95 are neighbours in reality, but
    359.9 degrees apart on the axis). The largest gap between points is the empty
    side; the cut goes just after it.

    Args:
        lon (np.ndarray): Query longitudes, already in the grid's own convention.
        origin (float): Current longitude-axis origin.

    Returns:
        float | None: The cut longitude in the grid's own frame, or None when the
            points are already contiguous.
    """
    offsets = np.sort(np.mod(np.asarray(lon, dtype=float) - origin, 360.0))
    if offsets.size < 2:
        return None
    gaps = np.diff(offsets)
    # the wrap-around gap, from the last point back to the first
    closing_gap = 360.0 - (offsets[-1] - offsets[0])
    if closing_gap >= gaps.max():
        # the largest gap already straddles the cut, so the points are contiguous
        return None
    return float(origin + offsets[int(gaps.argmax()) + 1])


def _roll_to_anchor(ds: "xr.Dataset", anchor: float) -> Tuple["xr.Dataset", bool]:
    """Re-cut a global grid's x axis at anchor, keeping it monotonic.

    Everything below the anchor moves to the top and gains 360, so the axis stays
    strictly increasing and slices normally. FES global grids carry both 0 and 360
    degrees, the same meridian twice, so the duplicate is dropped first; otherwise
    it lands twice on the re-cut axis and every later .sel fails.

    Args:
        ds (xr.Dataset): A global tide model dataset.
        anchor (float): Longitude to cut the axis at, in the grid's own frame.

    Returns:
        Tuple[xr.Dataset, bool]: The dataset, and whether it was re-cut. An anchor
            outside the axis leaves it alone and reports False, so the caller knows
            not to shift its query longitudes.
    """
    x = np.asarray(ds.coords["x"].values, dtype=float)
    if x.size > 1 and np.isclose((x[-1] - x[0]) % 360.0, 0.0):
        ds = ds.isel(x=slice(None, -1))
        x = np.asarray(ds.coords["x"].values, dtype=float)

    i = int(np.searchsorted(x, anchor))
    if i <= 0 or i >= x.size:
        return ds, False
    rolled = ds.roll(x=-i, roll_coords=False)
    rolled = rolled.assign_coords(x=np.concatenate([x[i:], x[:i] + 360.0]))
    rolled.attrs = dict(ds.attrs)
    return rolled, True


def _axis_slice(coord: np.ndarray, lo: float, hi: float) -> slice:
    """A slice covering [lo, hi], oriented to match the coordinate's direction.

    Args:
        coord (np.ndarray): The coordinate being sliced, ascending or descending.
        lo (float): Lower bound of the wanted range.
        hi (float): Upper bound of the wanted range.

    Returns:
        slice: Ready to pass to .sel on that coordinate.
    """
    if coord.size > 1 and coord[1] < coord[0]:
        return slice(hi, lo)
    return slice(lo, hi)


def _crop_halfwidths(
    lat_lo: float, lat_hi: float, dx: float, dy: float, reach_km: float
) -> Tuple[float, float]:
    """Calculate crop half-widths that cover a ground distance around query points.

    Latitude converts straight from km to degrees; longitude is scaled by
    cos(latitude) at the crop's poleward edge. Each axis gains two cells of padding.

    Args:
        lat_lo (float): Southernmost query latitude, in degrees.
        lat_hi (float): Northernmost query latitude, in degrees.
        dx (float): Grid spacing in longitude, in degrees.
        dy (float): Grid spacing in latitude, in degrees.
        reach_km (float): Ground distance the box must cover beyond the points.

    Returns:
        Tuple[float, float]: (longitude half-width, latitude half-width) in degrees.
            The longitude half-width is capped at 180, which is the whole axis.

    Example:
        The box still reaches 20 km where a degree of longitude is shortest:

        >>> lon_hw, lat_hw = _crop_halfwidths(74.9, 75.0, 1 / 16, 1 / 16, 20.0)
        >>> bool(lon_hw * _KM_PER_DEGREE * np.cos(np.radians(75.2)) >= 20.0)
        True
        >>> bool(lat_hw * _KM_PER_DEGREE >= 20.0)
        True
    """
    # cos(latitude) heads for zero at the poles which would cause an unbounded longitude buffer.
    _MAX_CROP_LATITUDE = 89.9
    lat_halfwidth = reach_km / _KM_PER_DEGREE + 2.0 * dy + 1e-6
    # Use the most poleward crop edge for longitude scaling.
    worst_lat = max(abs(lat_lo - lat_halfwidth), abs(lat_hi + lat_halfwidth))
    scale = np.cos(np.radians(min(worst_lat, _MAX_CROP_LATITUDE)))
    lon_halfwidth = reach_km / (_KM_PER_DEGREE * scale) + 2.0 * dx + 1e-6
    return float(min(lon_halfwidth, 180.0)), float(lat_halfwidth)


def _crop_to_points(
    ds: "xr.Dataset",
    lon: np.ndarray,
    lat: np.ndarray,
    extrapolate: bool = True,
    cutoff: float = 10.0,
) -> Tuple["xr.Dataset", np.ndarray]:
    """Crop a tide model grid around the requested points.

    The crop keeps enough padding for interpolation and, when enabled, for
    pyTMD's extrapolation search. Global grids are re-cut first when the points
    straddle the longitude seam. An infinite cutoff with extrapolation on skips
    the crop entirely, because the nearest wet cell could be anywhere.

    Args:
        ds (xr.Dataset): The full constituent grid, as opened by pyTMD.
        lon (np.ndarray): Query longitudes, already in the grid's own convention.
        lat (np.ndarray): Query latitudes in degrees.
        extrapolate (bool): Whether pyTMD will extrapolate off the wet grid, which
            is what the buffer makes room for.
        cutoff (float): Extrapolation cutoff in kilometres.

    Returns:
        Tuple[xr.Dataset, np.ndarray]: The cropped dataset, and the query
            longitudes, shifted if a global grid was re-cut.

    Raises:
        ValueError: If the requested points do not overlap the grid.
    """
    ds = _constituents_only(ds)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    x = np.asarray(ds.coords["x"].values, dtype=float)
    y = np.asarray(ds.coords["y"].values, dtype=float)

    if extrapolate and not np.isfinite(cutoff):
        logger.warning(
            "cutoff is infinite, so the nearest wet cell may lie anywhere and the "
            "whole grid has to be read; this needs ~16 GB of RAM on a global "
            "FES2022 download. Pass a finite cutoff in km to crop instead."
        )
        return _materialise(ds), lon

    # Add margin around pyTMD's extrapolation distance.
    _CROP_REACH_SAFETY = 2.0

    # Grid spacing is unchanged by any re-cut below, so the buffer can be sized now.
    dx = abs(x[1] - x[0]) if x.size > 1 else 0.0
    dy = abs(y[1] - y[0]) if y.size > 1 else 0.0
    reach_km = _CROP_REACH_SAFETY * float(cutoff) if extrapolate else 0.0
    lon_buffer, lat_buffer = _crop_halfwidths(
        float(np.nanmin(lat)), float(np.nanmax(lat)), dx, dy, reach_km
    )

    # Calculate crop padding from grid spacing.
    if _is_global_grid(x):
        anchor = _seam_anchor(lon, origin=float(np.nanmin(x)))
        if anchor is not None:
            # Preserve enough space for the western crop buffer.
            span = float(np.ptp(np.mod(np.asarray(lon) - anchor, 360.0)))
            backoff = min(lon_buffer, max(0.0, (360.0 - span) / 3.0))
            cut = anchor - backoff
            ds, was_rolled = _roll_to_anchor(ds, cut)
            x = np.asarray(ds.coords["x"].values, dtype=float)
            if was_rolled:
                # only lift the query points onto an axis that actually moved
                lon = np.where(lon < cut, lon + 360.0, lon)
                logger.info(
                    "region of interest straddles the model's seam; re-cut the grid "
                    "at %g deg",
                    cut,
                )

    cropped = ds.sel(
        x=_axis_slice(
            x, float(np.nanmin(lon)) - lon_buffer, float(np.nanmax(lon)) + lon_buffer
        ),
        y=_axis_slice(
            y, float(np.nanmin(lat)) - lat_buffer, float(np.nanmax(lat)) + lat_buffer
        ),
    )

    if cropped.sizes.get("x", 0) < 2 or cropped.sizes.get("y", 0) < 2:
        raise ValueError(
            f"the requested points (lon {np.nanmin(lon):.4f}..{np.nanmax(lon):.4f}, "
            f"lat {np.nanmin(lat):.4f}..{np.nanmax(lat):.4f}) do not overlap this tide "
            f"model, which covers lon {x.min():.4f}..{x.max():.4f}, "
            f"lat {y.min():.4f}..{y.max():.4f}"
        )

    cropped.attrs = dict(ds.attrs)
    return _materialise(cropped), lon


def _interp_constituents(
    ds: "xr.Dataset",
    X: "xr.DataArray",
    Y: "xr.DataArray",
    method: str = "linear",
    extrapolate: bool = True,
    cutoff: float = 10.0,
    partial_cell: bool = True,
) -> "xr.Dataset":
    """Interpolate harmonic constituents to the requested points.

    With partial_cell on, interpolation is renormalised over the wet corners of a
    partly masked cell; anything still missing falls back to pyTMD extrapolation.
    Basically use the closest corners of the tide model to cell to better predict
    the tide and ensure it works even if some of those corners do have valid tide data.

    Args:
        ds (xr.Dataset): The cropped constituent grid.
        X (xr.DataArray): Query x coordinates.
        Y (xr.DataArray): Query y coordinates.
        method (str): pyTMD interpolation method, 'linear' or 'nearest'.
        extrapolate (bool): Whether to fill anything still undefined by
            nearest-neighbour.
        cutoff (float): Extrapolation cutoff in kilometres.
        partial_cell (bool): Renormalise over the wet corners of a partly dry cell.

    Returns:
        xr.Dataset: One interpolated value per constituent per query point.
    """
    if not partial_cell:
        return ds.tmd.interp(
            X, Y, method=method, extrapolate=extrapolate, cutoff=cutoff
        )

    # constituents are individual sine waves that make the tide model
    names = list(ds.tmd.constituents)

    # Interpolate values and valid-cell weights separately.
    # numerator contains the actual tide values with NaNs filled in with 0
    numerator = ds.fillna(0)
    # copy metdata from ds into numerator and use dict so its gets its own copy of the metadata
    numerator.attrs = dict(ds.attrs)
    # weights are mask, 1 for tide data and 0 for missing tide data
    weights = ds.notnull().astype("float64")
    weights.attrs = dict(ds.attrs)

    # weighted sum of the available tide values
    num = numerator.tmd.interp(X, Y, method=method, extrapolate=False)
    # totall interpolation weights belonging to value data
    den = weights.tmd.interp(X, Y, method=method, extrapolate=False)

    # Normalize over available cell corners.
    # interpolate using available tide weights so bilinear interpolate better matches what tide data is available.
    local = num / den.where(
        den > 0.0
    )  # if all four corners were missing tides return a NaN
    local.attrs = dict(ds.attrs)
    for name in names:
        local[name].attrs = dict(ds[name].attrs)

    # Fill unresolved values with extrapolated values.
    if extrapolate:
        far = ds.tmd.interp(X, Y, method=method, extrapolate=True, cutoff=cutoff)
        for name in names:
            local[name] = local[name].where(local[name].notnull(), far[name])
    return local


def _tide_for_group(
    directory: Union[str, pathlib.Path],
    model: str,
    group: str,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    ts: "timescale.time.Timescale",
    method: str,
    extrapolate: bool,
    cutoff: float,
    partial_cell_interp: bool,
    corrections: str,
    infer_minor: bool,
    n_points: int,
    n_times: int,
) -> np.ndarray:
    """Predict one tide group (ocean_tide or load_tide) at the given points.

    Loads the group, crops it around the query points, interpolates, and evaluates
    it at each requested time.

    Args:
        directory (str | pathlib.Path): Folder holding the tide model.
        model (str): Tide model name in any accepted spelling.
        group (str): Constituent group to predict, 'ocean_tide' or 'load_tide'.
        lon (np.ndarray): Query longitudes in EPSG:4326.
        lat (np.ndarray): Query latitudes in EPSG:4326.
        ts (timescale.time.Timescale): Instants to predict at, shared by both groups.
        method (str): pyTMD interpolation method, 'linear' or 'nearest'.
        extrapolate (bool): Fill points off the wet grid by nearest-neighbour.
        cutoff (float): Extrapolation cutoff in kilometres.
        partial_cell_interp (bool): Renormalise over a partly dry cell's wet corners.
        corrections (str): pyTMD nodal-correction group.
        infer_minor (bool): Whether to add the inferred minor constituents.
        n_points (int): Number of query points, for the output shape.
        n_times (int): Number of instants, for the output shape.

    Returns:
        np.ndarray: Flat, point-major and time-minor, in meters.
    """
    _, ds = _tc()._open_tide_dataset(directory, model, group=group)

    # Match the model's longitude convention.
    group_lon = _match_lon_convention(ds, lon)
    # Crop the model around the query points.
    grid, group_lon = _crop_to_points(
        ds, group_lon, lat, extrapolate=extrapolate, cutoff=cutoff
    )

    X, Y = grid.tmd.coords_as(group_lon, lat, type="time series", crs=4326)
    local = _interp_constituents(
        grid,
        X,
        Y,
        method=method,
        extrapolate=extrapolate,
        cutoff=cutoff,
        partial_cell=partial_cell_interp,
    )

    # Evaluate the interpolated constituents at each prediction time.
    return _predict_blocked(
        local,
        ts.tide,
        ts.tt_ut1,
        corrections,
        infer_minor,
        n_points,
        n_times,
    )


def _as_point_major(pred: "xr.DataArray", n_points: int, n_times: int) -> np.ndarray:
    """Force a predicted DataArray onto an (n_points, n_times) array.

    'station' is pyTMD's name for the query-point axis. _tide_for_group asks for that
    layout by calling grid.tmd.coords_as(lon, lat, type="time series"), and pyTMD
    labels the resulting dimension 'station' the tide-gauge metaphor, one fixed location
    sampled at many instants. CoastSeg predicts one fixed point per transect, so here
    a station is a query point: station i is the seaward end of the i'th transect,
    at lon[i], lat[i].

    The dim order out of pyTMD is an implementation detail of how it broadcasts
    'station' against 'time', so transpose explicitly.

    Args:
        pred (xr.DataArray): One block of predicted tides from pyTMD.
        n_points (int): Expected number of query points, i.e. stations.
        n_times (int): Expected number of instants in this block.

    Returns:
        np.ndarray: Real-valued tides shaped (n_points, n_times).

    Raises:
        RuntimeError: If pyTMD returned a different shape, which would otherwise be
            silently reshaped into a wrong point/time pairing.
    """
    if "station" not in pred.dims:
        pred = pred.expand_dims("station")
    if "time" not in pred.dims:
        pred = pred.expand_dims("time")
    #  Ensure values are point-major (stations (aka transect points) as rows, time as columns)
    values = np.asarray(pred.transpose("station", "time").values)
    values = np.real(values).astype(float)
    if values.shape != (n_points, n_times):
        raise RuntimeError(
            f"pyTMD returned tides shaped {values.shape}, expected {(n_points, n_times)}"
        )
    return values


def _predict_blocked(
    local: "xr.Dataset",
    t: np.ndarray,
    deltat: np.ndarray,
    corrections: str,
    infer_minor: bool,
    n_points: int,
    n_times: int,
) -> np.ndarray:
    """Predict tides in blocks of time so the temporaries stay bounded.

    pyTMD builds (points, constituents, times) intermediates, so a whole ROI at once
    can reach hundreds of MB. Interpolation already ran once outside this loop; only
    the cheap harmonic summation is blocked.

    Args:
        local (xr.Dataset): Harmonic constants already interpolated to the points.
        t (np.ndarray): Instants to predict at, as pyTMD tide-days.
        deltat (np.ndarray): TT - UT1 at each instant, or a single value to repeat.
        corrections (str): pyTMD nodal-correction group; see `model_tides`.
        infer_minor (bool): Whether to add the inferred minor constituents.
        n_points (int): Number of query points.
        n_times (int): Number of instants.

    Returns:
        np.ndarray: Flat, point-major and time-minor: point p at time t is at index
            p * n_times + t, matching pyTMD 2's tile/repeat pairing.
    """
    n_cons = max(1, len(local.tmd.constituents))
    per_time = max(1, n_points * n_cons)
    block = int(max(1, min(n_times, _MAX_PREDICT_ELEMENTS // per_time)))

    t = np.atleast_1d(np.asarray(t, dtype=float))
    deltat = np.atleast_1d(np.asarray(deltat, dtype=float))
    if deltat.size == 1:
        deltat = np.repeat(deltat, n_times)

    out = np.empty((n_points, n_times), dtype=float)
    for start in range(0, n_times, block):
        stop = min(start + block, n_times)
        tb, db = t[start:stop], deltat[start:stop]
        pred = local.tmd.predict(tb, deltat=db, corrections=corrections)
        if infer_minor:
            pred = pred + local.tmd.infer(
                tb, deltat=db, corrections=corrections, minor=None
            )
        out[:, start:stop] = _as_point_major(pred, n_points, stop - start)
    return out.reshape(-1)


def _dates_for_transect(
    timeseries_df: pd.DataFrame, transect_id: Union[str, int]
) -> Optional[np.ndarray]:
    """Return dates associated with a transect.

    Handles both wide matrices and long-form frames. Transect IDs compare as
    strings, so numeric and string identifiers still match.

    Args:
        timeseries_df (pd.DataFrame): A wide (dates x transect_id) matrix or a
            long-form frame.
        transect_id (str | int): The transect to select, or '' for every date in the
            frame.

    Returns:
        np.ndarray | None: The transect's dates, or None when the wide matrix has no
            column for it.
    """
    if transect_id == "":
        return timeseries_df[["dates"]].dropna()["dates"].values

    key = str(transect_id)
    if "transect_id" in timeseries_df.columns:
        matching_rows = timeseries_df[timeseries_df["transect_id"].astype(str) == key]
        return matching_rows[["dates"]].dropna()["dates"].values

    column = next((c for c in timeseries_df.columns if str(c) == key), None)
    if column is None:
        return None
    return timeseries_df[["dates", column]].dropna()["dates"].values


def _batch_by_date_signature(
    members: List[tuple],
) -> List[Tuple[List[tuple], np.ndarray]]:
    """Group transects that need exactly the same dates. Cheap, and exact.

    Args:
        members (List[tuple]): Transect records, see `_predict_region` for the
            tuple layout.

    Returns:
        List[Tuple[List[tuple], np.ndarray]]: One (members, dates) batch per distinct
            date set.
    """
    buckets: Dict[bytes, List[tuple]] = {}
    for member in members:
        dates = np.ascontiguousarray(np.unique(member[3]))
        signature = hashlib.blake2b(dates.tobytes(), digest_size=16).digest()
        buckets.setdefault(signature, []).append(member)
    return [
        (group, np.unique(np.concatenate([m[3] for m in group])))
        for group in buckets.values()
    ]


def _split_oversized_batches(
    batches: List[Tuple[List[tuple], np.ndarray]],
) -> List[Tuple[List[tuple], np.ndarray]]:
    """Chunk any batch whose points x times block would exceed _MAX_BATCH_CELLS.

    Splitting by transect rather than by date keeps every batch's date set intact, so
    each transect is still looked up in one table.

    Args:
        batches (List[Tuple[List[tuple], np.ndarray]]): (members, dates) batches.

    Returns:
        List[Tuple[List[tuple], np.ndarray]]: The same work in blocks within the cap.
            A transect that exceeds it on its own is left alone; nothing left to split.
    """
    out: List[Tuple[List[tuple], np.ndarray]] = []
    for group, times in batches:
        per_transect = max(1, len(times))
        limit = max(1, int(_MAX_BATCH_CELLS // per_transect))
        if len(group) <= limit:
            out.append((group, times))
            continue
        logger.info(
            "splitting a %d transect x %d date block into chunks of %d transects to "
            "stay under %d cells",
            len(group),
            len(times),
            limit,
            _MAX_BATCH_CELLS,
        )
        for start in range(0, len(group), limit):
            out.append((group[start : start + limit], times))
    return out


def _predict_region(
    region_dir: str, members: List[tuple], config: dict
) -> List[Tuple[str, pd.DataFrame]]:
    """Predict tides for every transect in one region with a single model read.

    Args:
        region_dir (str): Folder holding the tide model for this batch.
        members (List[tuple]): One record per transect, each
            (transect_id, x, y, dates, directory), where x/y are the seaward
            point in EPSG:4326 and dates is that transect's own datetime64 array.
        config (dict): Prediction settings; see `setup_tide_model_config`.

    Returns:
        List[Tuple[str, pd.DataFrame]]: One (transect_id, frame) pair per transect,
            each frame holding only that transect's dates.
    """
    per_transect = [m[3] for m in members]
    union = np.unique(np.concatenate(per_transect))
    cost = len(members) * len(union)
    needed = sum(len(d) for d in per_transect)

    # determine how many tide predictions will be thrown away because there was no date in the timeseries for them at that location
    waste = cost - needed

    # The cheap comparison comes first, so the common case never pays for the hashing.
    batches = [(members, union)]
    if cost > _MAX_BATCH_CELLS or waste > _WASTE_CELLS_PER_EXTRA_CALL:
        # organize by unqiue dates
        buckets = _batch_by_date_signature(members)
        extra_calls = len(buckets) - 1
        if extra_calls > 0 and (
            cost > _MAX_BATCH_CELLS or waste > extra_calls * _WASTE_CELLS_PER_EXTRA_CALL
        ):
            logger.info(
                "date union wastes %d of %d cells across %d transects; %d extra "
                "call(s) is the cheaper trade, grouping by date set",
                waste,
                cost,
                len(members),
                extra_calls,
            )
            batches = buckets

    batches = _split_oversized_batches(batches)

    frames: List[Tuple[str, pd.DataFrame]] = []
    # for each batch of dates and times get the tide model predictions
    for group, times in batches:
        xs = np.array([g[1] for g in group], dtype=float)
        ys = np.array([g[2] for g in group], dtype=float)
        tide_df = model_tides(
            xs,
            ys,
            times,
            model=config["MODEL"],
            directory=region_dir,
            epsg=config.get("EPSG", 4326),
            method=config.get("METHOD", "bilinear"),
            extrapolate=config.get("EXTRAPOLATE", True),
            cutoff=config.get("CUTOFF", 10.0),
        )
        # model_tides returns rows point-major and time-minor
        matrix = np.asarray(tide_df["tide"].values, dtype=float).reshape(
            len(group), len(times)
        )
        index = pd.DatetimeIndex(times)
        for i, (tid, px, py, dates, _) in enumerate(group):
            pos = index.get_indexer(pd.DatetimeIndex(dates))
            tide = np.full(len(dates), np.nan, dtype=float)
            found = pos >= 0
            tide[found] = matrix[i, pos[found]]
            frames.append(
                (
                    tid,
                    pd.DataFrame(
                        {
                            "dates": pd.to_datetime(dates, utc=True),
                            "x": np.repeat(px, len(dates)),
                            "y": np.repeat(py, len(dates)),
                            "tide": tide,
                            "transect_id": tid,
                        }
                    ),
                )
            )
    return frames


def _region_folder(prefix: str, region_id: object) -> Optional[str]:
    """The clipped model folder a region_id names, e.g. '<prefix>3'.

    LEGACY: only used for the clipped region0..region10 layout.

    Args:
        prefix (str): Path prefix the region number is appended to, i.e. the
            REGION_DIRECTORY of the tide model config.
        region_id (object): The region the point joined to. Arrives as a float, not
            an int, whenever any point in the batch failed the join and put a NaN in
            the column.

    Returns:
        str | None: The folder to read, or None when the point was never assigned a
            region.

    Example:
        >>> _region_folder("tide_model/region", 3.0)
        'tide_model/region3'
        >>> _region_folder("tide_model/region", float("nan")) is None
        True
    """
    if region_id is None:
        return None
    try:
        number = float(region_id)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return f"{prefix}{int(number)}"


def predict_tides_for_df(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Predict tides for each transect and its associated dates.

    Predictions are batched by model directory to avoid repeated model reads. Each
    transect is evaluated at its seaward point, on its own dates only.

    Args:
        seaward_points_gdf (gpd.GeoDataFrame): One seaward point per transect.
        timeseries_df (pd.DataFrame): Time series data for each transect.
        config (dict): Tide model configuration. Must contain the keys:

            - REGION_DIRECTORY: full path to the FES model region folder
            - MODEL: The tide model to use. Defaults to 'FES2022'
            - LAYOUT: optional, 'global' to read the un-clipped model at "DIRECTORY"
              instead of the per-region folders. Defaults to 'regions'.

    Returns:
        pd.DataFrame: Columns 'dates', 'x', 'y', 'tide' and 'transect_id' - one block
            of rows per transect, in seaward_points_gdf order, carrying only that
            transect's own dates.
    """
    region_directory = config["REGION_DIRECTORY"]
    is_global = config.get("LAYOUT") == "global"
    global_directory = config.get("DIRECTORY")

    # Keep one seaward point per transect.
    duplicated = seaward_points_gdf["transect_id"].duplicated()
    if duplicated.any():
        logger.warning(
            "%d seaward point(s) repeat a transect_id and were ignored; a transect "
            "has one seaward point. Repeated ids: %s",
            int(duplicated.sum()),
            sorted({str(t) for t in seaward_points_gdf.loc[duplicated, "transect_id"]}),
        )
        seaward_points_gdf = seaward_points_gdf[~duplicated]

    members: List[tuple] = []
    order: Dict[str, int] = {}
    unassigned: List[str] = []
    # Get the dates for each unique seaward point to predict tides for
    for _, row in seaward_points_gdf.iterrows():
        transect_id = row["transect_id"]
        dates = _dates_for_transect(timeseries_df, transect_id)
        if dates is None or len(dates) == 0:
            continue
        if is_global:
            # use unclipped tide model so no need use regions file
            directory = global_directory
        else:
            # get the region that seaward point lies within
            region = _region_folder(region_directory, row.get("region_id"))
            if region is None:
                unassigned.append(str(transect_id))
                continue
            directory = region
        order.setdefault(transect_id, len(order))
        members.append(
            (
                transect_id,
                float(row.geometry.x),
                float(row.geometry.y),
                np.asarray(dates),
                directory,
            )
        )

    if unassigned:
        logger.warning(
            "%d transect(s) fall outside every tide region and were left without a "
            "tide; their observations come through with a NaN cross_distance: %s",
            len(unassigned),
            sorted(unassigned)[:20],
        )

    # Return an empty result when no transects match the time series.
    if not members:
        # get all the wanted transect ids
        wanted = [str(t) for t in seaward_points_gdf["transect_id"].tolist()]
        # get all the available transect ids in the timeseries and read transect ids regardless of timeseries df layout
        available = (
            sorted({str(t) for t in timeseries_df["transect_id"]})
            if "transect_id" in timeseries_df.columns
            else [
                str(c) for c in timeseries_df.columns if c != "dates"
            ]  # the else is for if each transect id is a column name eg CoastSat default layout
        )
        logger.warning(
            "No transect in the tide request matched the time series, so no tides "
            "were predicted. Looked for %s; the time series holds %s",
            wanted[:10],
            available[:10],
        )
        # dates dtype is taken from the caller so the merge key matches its resolution exactly.
        dates_dtype = (
            timeseries_df["dates"].dtype
            if "dates" in timeseries_df.columns
            else "datetime64[ns, UTC]"
        )
        return pd.DataFrame(
            {
                "dates": pd.Series(dtype=dates_dtype),
                "x": pd.Series(dtype="float64"),
                "y": pd.Series(dtype="float64"),
                "tide": pd.Series(dtype="float64"),
                "transect_id": pd.Series(dtype="object"),
            }
        )

    # Group transects by model directory for batched prediction.
    by_region: Dict[str, List[tuple]] = {}
    for member in members:
        by_region.setdefault(member[4], []).append(member)

    tagged: List[Tuple[str, pd.DataFrame]] = []
    with tqdm(
        total=len(by_region),
        desc=f"  Predicting tides for {len(members)} transects",
    ) as progress:
        for region_dir, group in by_region.items():
            tagged.extend(_predict_region(region_dir, group, config))
            progress.update(1)

    # Restore the caller's transect order, which the per-region grouping broke up.
    tagged.sort(key=lambda item: order[item[0]])
    return pd.concat([frame for _, frame in tagged])


def model_tides(
    x: Union[float, Collection[float], np.ndarray],
    y: Union[float, Collection[float], np.ndarray],
    time: Union[np.ndarray, pd.DatetimeIndex, pd.Timestamp],
    transect_id: str = "",
    model: str = "FES2022",
    directory: Union[str, pathlib.Path, None] = None,
    epsg: Union[int, str] = 4326,
    method: str = "bilinear",
    extrapolate: bool = True,
    cutoff: float = 10.0,
    *,
    corrections: str = "FES",
    infer_minor: bool = True,
    partial_cell_interp: Optional[bool] = None,
    include_load_tide: Optional[bool] = None,
    group: str = "ocean_tide",
) -> pd.DataFrame:
    """Compute tides at points and times using tidal harmonics.

    Supports the FES2014 and FES2022 Finite Element Solution models, read through
    `pyTMD` 3. Given several x, y points, every point is evaluated at every timestep.

    Args:
        x (float | list): One or more x coordinates. Lat/lon by default; use `epsg`
            for a custom coordinate reference system.
        y (float | list): One or more y coordinates, paired with `x`.
        time (np.ndarray | pd.DatetimeIndex): The times to model tides at, in UTC, as
            'datetime64[ns]' values or a 'pandas.DatetimeIndex'.
        transect_id (str, optional): The transect the points belong to. When given, a
            'transect_id' column is added to the result. Defaults to "".
        model (str, optional): "FES2022" (the CoastSeg default) or "FES2014".
            Defaults to "FES2022".
        directory (str, optional): The directory containing tide model data files, e.g.
            one of the clipped region folders:

            - {directory}/fes2022b/ocean_tide/
            - {directory}/fes2014/ocean_tide/

            Both the plain `ocean_tide` folder and AVISO's dated variants
            (`ocean_tide_20241025`) are accepted. Required; defaults to None.
        epsg (int, optional): Coordinate system of 'x' and 'y'. Defaults to 4326.
        method (str, optional): Interpolation method. pyTMD 3 implements only 'linear'
            and 'nearest'; the pyTMD 2 spellings 'bilinear' and 'spline' are still
            accepted and map onto 'linear'. Defaults to "bilinear".
        extrapolate (bool, optional): Fill locations outside the model domain by
            nearest-neighbour. Defaults to True.
        cutoff (int | float, optional): Extrapolation cutoff in kilometers. A finite
            cutoff crops the model to a box covering it in every direction at the
            site's own latitude. `np.inf` extrapolates everywhere, as in pyTMD, which
            disables the crop and reads the whole model: ~16 GB of RAM on a global
            FES2022 download. Defaults to 10.0.
        corrections (str, optional): pyTMD nodal-correction group, i.e. which family of
            formulae corrects each constituent for the 18.6-year precession of the
            Moon's orbit. Leave it alone; it is a parameter only so the pyTMD 2.x
            migration tests can pin the old behaviour. Defaults to 'FES', which is
            correct for both models CoastSeg supports.
        infer_minor (bool, optional): Add the inferred minor constituents. Defaults to
            True, as in pyTMD 2.
        partial_cell_interp (bool, optional): Renormalise over the wet corners of a
            partly dry cell, as pyTMD 2 did. Defaults to TIDE_PARTIAL_CELL_INTERP.
        include_load_tide (bool, optional): Add the ocean loading tide so the result is
            the geocentric tide, matching CoastSat. Pass False for the ocean tide
            alone, e.g. against a tide gauge. Defaults to TIDE_INCLUDE_LOAD_TIDE.
        group (str, optional): The base tide group, 'ocean_tide' or 'load_tide'. The
            load tide is only added on top when this is 'ocean_tide'. Defaults to
            "ocean_tide".

    Returns:
        pd.DataFrame: Tide heights in **meters**, with the columns 'dates', 'x', 'y',
            'tide' and, when a transect_id was given, 'transect_id'. Rows are
            point-major and time-minor: point p at time t is at index p * n_times + t.

    Raises:
        ValueError: If no `directory` was given, or `method` is unknown.
        FileNotFoundError: If `directory` does not exist.
    """
    # Check tide directory is accessible
    if directory is None:
        raise ValueError("model_tides requires the path to a tide model directory")
    directory = pathlib.Path(directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError("Invalid tide directory")

    # Validate input arguments
    if method not in _METHOD_ALIASES:
        raise ValueError(
            f"Unknown interpolation method {method!r}; "
            f"expected one of {sorted(_METHOD_ALIASES)}"
        )
    interp_method = _METHOD_ALIASES[method]
    if method != interp_method:
        logger.info(
            "pyTMD 3 removed %r interpolation; using %r instead", method, interp_method
        )

    # If time passed as a single Timestamp, convert to datetime64
    if isinstance(time, pd.Timestamp):
        time = time.to_datetime64()

    # Handle numeric or array inputs
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    time = np.atleast_1d(time)

    # Determine point and time counts
    assert len(x) == len(y), "x and y must be the same length"
    n_points = len(x)
    n_times = len(time)

    # Converting x,y from EPSG to latitude/longitude
    try:
        # EPSG projection code string or int
        crs1 = pyproj.CRS.from_epsg(int(epsg))
    except (ValueError, pyproj.exceptions.CRSError):
        # Projection SRS string
        crs1 = pyproj.CRS.from_string(epsg)

    # Output coordinate reference system
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    lon, lat = transformer.transform(x.flatten(), y.flatten())
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))

    # Read at call time: 8_predict_tides_from_csv.py sets
    # tide_correction.TIDE_INCLUDE_LOAD_TIDE before predicting.
    if partial_cell_interp is None:
        partial_cell_interp = _tc().TIDE_PARTIAL_CELL_INTERP
    if include_load_tide is None:
        include_load_tide = _tc().TIDE_INCLUDE_LOAD_TIDE

    # Convert datetime
    ts = timescale.from_datetime(time.flatten())

    predict_kwargs = dict(
        lon=lon,
        lat=lat,
        ts=ts,
        method=interp_method,
        extrapolate=extrapolate,
        cutoff=cutoff,
        partial_cell_interp=partial_cell_interp,
        corrections=corrections,
        infer_minor=infer_minor,
        n_points=n_points,
        n_times=n_times,
    )
    tide = _tide_for_group(directory, model, group, **predict_kwargs)

    # Add the solid-earth loading term so the result is the geocentric tide, which
    # is what CoastSat reports (SDS_slope.compute_tide sums the 'tide' and
    # 'radial' handlers). Only meaningful when the ocean tide was the base group.
    if include_load_tide and group == "ocean_tide":
        try:
            tide = tide + _tide_for_group(
                directory, model, "load_tide", **predict_kwargs
            )
        except FileNotFoundError as exc:
            raise _tc()._load_tide_missing_error(exc) from exc

    columns = {
        "dates": np.tile(time, n_points),
        "x": np.repeat(x, n_times),
        "y": np.repeat(y, n_times),
        "tide": tide,
    }
    # if a transect id was passed, orient the result by it
    if transect_id:
        columns["transect_id"] = transect_id

    df = pd.DataFrame(columns)
    df["dates"] = pd.to_datetime(df["dates"], utc=True)
    return df


# ---------------------------------------------------------------------------
# LEGACY: clipped region0..region10 support.
#
# Everything from here to predict_tides() exists only to serve tide models that
# were split into regional folders by download_tide_model.clip_model_to_regions.
# The un-clipped model is the default (see resolve_model_layout) and needs none
# of it: no region geojson, no spatial join, no region_id.
#
# SUPPORT FOR THE CLIPPED LAYOUT MAY BE REMOVED IN A FUTURE VERSION OF COASTSEG.
# It is kept so existing installs keep working without re-downloading the model.
# ---------------------------------------------------------------------------


def load_regions_from_geojson(geojson_path: str) -> gpd.GeoDataFrame:
    """Load regions from a GeoJSON file and assign a region_id based on index.

    LEGACY: only used for the clipped region0..region10 layout, which may be removed
    in a future version of CoastSeg.

    Args:
        geojson_path (str): Path to the GeoJSON file containing regions.

    Returns:
        gpd.GeoDataFrame: The regions, with a 'region_id' column added.
    """
    gdf = gpd.read_file(geojson_path)
    gdf["region_id"] = gdf.index
    return gdf


def perform_spatial_join(
    seaward_points_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Join seaward points onto the regions that contain them.

    Both GeoDataFrames must be in CRS 4326, or the join matches nothing.

    LEGACY: only used for the clipped region0..region10 layout, which may be removed
    in a future version of CoastSeg. The un-clipped model needs no region lookup.

    tide_regions_map.geojson stores regions 0 and 6 below -180 (down to -310) and
    region 7 above +180, while transect points are in the standard -180..180 frame,
    so a plain Cartesian intersection misses them by exactly 360 degrees: a transect
    just west of the dateline gets no region_id, and the caller then fails on a
    'region<nan>' directory. Points that miss the first pass are retried shifted east
    and west.

    Args:
        seaward_points_gdf (gpd.GeoDataFrame): Seaward points, in CRS 4326.
        regions_gdf (gpd.GeoDataFrame): Tide model regions, in CRS 4326.

    Returns:
        gpd.GeoDataFrame: The points with a 'region_id' column, NaN where unmatched.
    """
    joined_gdf = gpd.sjoin(
        seaward_points_gdf, regions_gdf, how="left", predicate="intersects"
    )
    joined_gdf.drop(columns="index_right", inplace=True)

    if "region_id" not in joined_gdf.columns:
        return joined_gdf

    unmatched = joined_gdf["region_id"].isna()
    if not unmatched.any():
        return joined_gdf

    # The retry maps rows back by index, which is only well defined when the caller's
    # index is unique. It always is for the seaward points CoastSeg builds.
    if not seaward_points_gdf.index.is_unique:
        logger.warning(
            "seaward points have a non-unique index; skipping the antimeridian retry"
        )
        return joined_gdf

    for shift in (360.0, -360.0):
        if not unmatched.any():
            break
        # A point that failed to match produces exactly one all-NaN row, so the
        # unmatched index values are unique even if other points matched twice.
        missing_index = joined_gdf.index[unmatched].unique()
        retry = seaward_points_gdf.loc[missing_index].copy()
        retry["geometry"] = retry.geometry.translate(xoff=shift)

        found = gpd.sjoin(retry, regions_gdf, how="left", predicate="intersects")
        resolved = found["region_id"].dropna()
        if resolved.empty:
            continue
        # keep the first match if a shifted point lands in overlapping regions
        resolved = resolved[~resolved.index.duplicated(keep="first")]

        joined_gdf.loc[resolved.index, "region_id"] = resolved
        logger.info(
            "assigned %d transect point(s) to a tide region after shifting their "
            "longitude by %+g degrees",
            len(resolved),
            shift,
        )
        unmatched = joined_gdf["region_id"].isna()

    if unmatched.any():
        logger.warning(
            "%d transect point(s) fall outside every tide region, even allowing for "
            "the antimeridian: %s",
            int(unmatched.sum()),
            [
                (round(geom.x, 4), round(geom.y, 4))
                for geom in joined_gdf.loc[unmatched, "geometry"]
            ],
        )

    return joined_gdf


def predict_tides(
    transects_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    model_regions_geojson_path: str,
    config: dict,
) -> pd.DataFrame:
    """Predict tides based on input data and configurations.

    Args:
        transects_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the transect data.
        timeseries_df (pd.DataFrame): Raw time series data for each transect.
        model_regions_geojson_path (str): Path to the model regions GeoJSON.
        config (dict): Tide model configuration, typically holding:

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

    Returns:
        pd.DataFrame: The predicted tides; see `predict_tides_for_df`.
    """
    # Get the seaward points in CRS 4326
    seaward_points_gdf = common.get_seaward_points_gdf(transects_gdf)

    # Default the un-clipped model covers the whole planet, so no region to look up.
    if config.get("LAYOUT") == "global":
        logger.info("using the un-clipped tide model; skipping the region lookup")
        return predict_tides_for_df(seaward_points_gdf, timeseries_df, config)

    # LEGACY: the clipped region0..region10 layout.
    # Everything below exists only to map each transect onto the regional grid that contains it, and may be removed
    # in a future version of CoastSeg.
    # Read in the model regions from a GeoJSON file
    regions_gdf = load_regions_from_geojson(model_regions_geojson_path)
    # convert to crs 4326 if it is not already
    if regions_gdf.crs is None:
        regions_gdf = regions_gdf.set_crs("epsg:4326")
    else:
        regions_gdf = regions_gdf.to_crs("epsg:4326")
    # Perform a spatial join to get the region_id for each point in seaward_points_gdf
    regional_seaward_points_gdf = perform_spatial_join(seaward_points_gdf, regions_gdf)
    # predict_tides_for_df groups its own members by model directory, so the region
    # split that used to happen here (handle_tide_predictions -> model_tides_by_region_id)
    # only changed the number of calls, and cost the caller's transect order on the way.
    return predict_tides_for_df(regional_seaward_points_gdf, timeseries_df, config)
