"""Frame builders shared by the tide-correction tests.

Every tide test needs one of three shapes: a long time series (dates x transect_id
rows), the wide matrix CoastSeg pivots it into, or a GeoDataFrame of seaward points.
They used to be rebuilt in five separate files under five different names, so a
change to the frame CoastSeg actually passes around had to be mirrored five times.

These are plain factories rather than pytest fixtures because callers parameterise
them; import them directly.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def long_timeseries(transect_ids, dates, **columns):
    """A long time series: one row per (date, transect).

    Args:
        transect_ids: transect id per row.
        dates: date per row; parsed to UTC.
        **columns: any further columns, each a scalar broadcast over every row or a
            sequence one value per row. cross_distance=10.0 and tide=0.5 are
            the usual ones.

    Example:
        >>> long_timeseries(["t1", "t1"], ["2021-01-01", "2021-01-02"], tide=0.5)
    """
    frame = pd.DataFrame(
        {
            "transect_id": list(transect_ids),
            "dates": pd.to_datetime(list(dates), utc=True),
        }
    )
    for name, value in columns.items():
        frame[name] = value
    return frame


def matrix_timeseries(dates, columns):
    """The wide form CoastSeg feeds in: a dates column plus one column per transect.

    columns maps transect_id -> list of values, with None where that transect
    has no observation on that date.
    """
    data = {"dates": pd.to_datetime(dates, utc=True)}
    data.update({tid: pd.Series(vals, dtype="float64") for tid, vals in columns.items()})
    return pd.DataFrame(data)


def points_gdf(entries):
    """Seaward points, as (transect_id, lon, lat) or (transect_id, region_id,
    lon, lat) tuples.

    A region_id column is only added when the entries carry one, since the global
    layout has to work without it.
    """
    entries = list(entries)
    with_region = bool(entries) and len(entries[0]) == 4

    data = {"transect_id": [e[0] for e in entries]}
    if with_region:
        data["region_id"] = [e[1] for e in entries]
    geometry = [Point(e[-2], e[-1]) for e in entries]

    return gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
