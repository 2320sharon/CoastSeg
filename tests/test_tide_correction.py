import inspect
import json
import os
import pathlib
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from coastseg import download_tide_model as dtm
from coastseg import tide_correction as tc
from coastseg import tide_predict as tp
from coastseg.tide_correction import load_regions_from_geojson, save_transect_settings
from tests.tide_helpers import long_timeseries


@pytest.mark.parametrize(
    "label, actual, expected",
    [
        # predicted_tides.csv and the corrected matrix are pivoted from the same long
        # frame, so a transect intersecting twice on one date must collapse the same
        # way in both.
        ("matrix aggfunc", lambda: tc.TIDE_MATRIX_AGGFUNC, "mean"),
        # choosing a layout is opt-in: unset still means un-clipped with a fallback
        (
            "tide model layout",
            lambda: inspect.signature(tc.correct_all_tides)
            .parameters["tide_model_layout"]
            .default,
            "auto",
        ),
        # CoastSat reports ocean + load, so CoastSeg does too
        ("include load tide", lambda: tc.TIDE_INCLUDE_LOAD_TIDE, True),
        (
            "nodal corrections",
            lambda: inspect.signature(tc.model_tides).parameters["corrections"].default,
            "FES",
        ),
        # every group aviso_fes_list requests needs a fallback, or a listing hiccup
        # abandons a multi-hour download
        (
            "download fallbacks",
            lambda: set(dtm.FES2022_FALLBACK_DIRS),
            {"ocean_tide", "load_tide", "ocean_tide_extrapolated"},
        ),
    ],
)
def test_shipped_default(label, actual, expected):
    assert actual() == expected, f"the shipped {label} default changed"


@pytest.mark.parametrize(
    "existing",
    [{"reference_elevation": 0, "beach_slope": 0}, {}, None],
    ids=["overwrites-existing-values", "empty-file", "no-file-yet"],
)
def test_save_transect_settings(existing):
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_file = os.path.join(tmpdir, "transects_settings.json")
        if existing is not None:
            with open(settings_file, "w") as f:
                json.dump(existing, f)

        save_transect_settings(tmpdir, 1.23, 4.56)

        with open(settings_file, "r") as f:
            settings = json.load(f)
        assert settings["reference_elevation"] == 1.23
        assert settings["beach_slope"] == 4.56
        # nothing names a tide model unless one was actually read; see
        # test_a_failed_correction_records_no_tide_model
        assert not any(key.startswith("tide_model") for key in settings)


def test_load_regions_from_geojson():
    # Create a temporary GeoJSON file
    with tempfile.NamedTemporaryFile(suffix=".geojson") as tmp:
        geojson_path = tmp.name

        # Create a GeoDataFrame with some test data
        regions_gdf = gpd.GeoDataFrame(
            {
                "region_id": [1, 2, 3],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            }
        )

        # Save the GeoDataFrame to the temporary GeoJSON file
        regions_gdf.to_file(geojson_path, driver="GeoJSON")

        # Call the function to load the regions
        loaded_regions_gdf = load_regions_from_geojson(geojson_path)

        # Check that the loaded regions is a GeoDataFrame
        assert isinstance(loaded_regions_gdf, gpd.GeoDataFrame)

        # Check that the 'region_id' column is added
        assert "region_id" in loaded_regions_gdf.columns

        # Check that the number of regions is correct
        assert len(loaded_regions_gdf) == 3


# --------------------------------------------------------------------------
# the steps correct_tides orchestrates
#
# Splitting correct_tides made these individually reachable; before, the only way
# to exercise any of them was to run the whole correction against the tide model.
# Nothing here needs pyTMD or the model.
# --------------------------------------------------------------------------


def _observations():
    return long_timeseries(
        ["t1", "t1", "t2", "t2"],
        ["2021-01-01", "2021-01-02"] * 2,
        cross_distance=[10.0, 11.0, 20.0, 21.0],
    )


def test_attach_slopes_from_a_single_value():
    result = tc._attach_slopes(_observations(), 0.02)

    assert (result["slope"] == 0.02).all()


def test_attach_slopes_from_a_file(tmp_path):
    path = tmp_path / "slopes.csv"
    pd.DataFrame(
        {
            "dates": ["2021-01-01", "2021-01-02"],
            "transect_id": ["t1", "t1"],
            "slope": [0.02, 0.04],
        }
    ).to_csv(path, index=False)

    result = tc._attach_slopes(_observations(), str(path))

    assert result.loc[result["transect_id"] == "t1", "slope"].tolist() == [0.02, 0.04]
    # t2 is absent from the file, so it has no slope  nothing is substituted, and
    # its cross_distance comes out NaN rather than fabricated
    assert result.loc[result["transect_id"] == "t2", "slope"].isna().all()


def test_resolve_tides_from_a_file(tmp_path):
    path = tmp_path / "tides.csv"
    pd.DataFrame(
        {
            "dates": ["2021-01-01", "2021-01-02"],
            "transect_id": ["t1", "t1"],
            "tide": [0.5, 0.6],
        }
    ).to_csv(path, index=False)

    result = tc._resolve_tides(
        _observations(), gpd.GeoDataFrame(), roi_id="roi_1", tides_file=str(path)
    )

    assert result.loc[result["transect_id"] == "t1", "tide"].tolist() == [0.5, 0.6]


def test_resolve_tides_reports_a_missing_file():
    with pytest.raises(FileNotFoundError, match="Tide CSV file not found"):
        tc._resolve_tides(
            _observations(),
            gpd.GeoDataFrame(),
            roi_id="roi_1",
            tides_file="nowhere/tides.csv",
        )


def test_uncorrectable_observations_are_kept_and_come_through_as_nan():
    """Regression: these used to be dropped, which made "no shoreline detected" and
    "shoreline detected but not correctable" look identical in the output."""
    frame = _observations().assign(
        tide=[0.5, np.nan, 0.7, 0.8],
        slope=[0.02, 0.02, np.nan, 0.02],
        reference_elevation=0.0,
    )

    corrected = tc.apply_tide_correction_df(frame.copy())

    # nothing is removed; the two observations that could not be corrected are
    # present with a NaN cross_distance
    assert len(corrected) == len(frame)
    assert corrected["cross_distance"].isna().sum() == 2


def test_save_corrected_outputs_writes_every_file(tmp_path):
    corrected = _observations().assign(tide=0.5, slope=0.02, reference_elevation=0.0)
    transects = gpd.GeoDataFrame(
        {"id": ["t1", "t2"]},
        geometry=[
            LineString([(-75.60, 36.10), (-75.59, 36.11)]),
            LineString([(-75.50, 36.20), (-75.49, 36.21)]),
        ],
        crs="epsg:4326",
    )

    merged = tc._save_corrected_outputs(corrected, transects, str(tmp_path))

    for name in (
        "predicted_tides.csv",
        "tidally_corrected_transect_time_series.csv",
        "tidally_corrected_transect_time_series_merged.csv",
    ):
        assert (tmp_path / name).is_file(), f"{name} was not written"
    assert len(merged) == len(corrected)
    # the matrix keeps both transects and sorts them
    matrix = pd.read_csv(tmp_path / "tidally_corrected_transect_time_series.csv")
    assert list(matrix.columns) == ["dates", "t1", "t2"]


@pytest.fixture
def stub_predictions(monkeypatch):
    """Predict a constant tide, and read the un-clipped layout without a model."""
    monkeypatch.setattr(tc, "resolve_model_layout", lambda *a, **k: "global")

    def _stub(x, y, time, transect_id="", **kwargs):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        time = np.atleast_1d(time)
        return pd.DataFrame(
            {
                "dates": np.tile(time, len(x)),
                "x": np.repeat(x, len(time)),
                "y": np.repeat(y, len(time)),
                "tide": np.full(len(x) * len(time), 1.5),
            }
        )

    monkeypatch.setattr(tp, "model_tides", _stub)


def _one_transect_gdf(ids=("t1",)):
    return gpd.GeoDataFrame(
        {"id": list(ids), "type": ["transect"] * len(ids)},
        geometry=[
            LineString([(-120.5 + i * 0.1, 35.2), (-120.45 + i * 0.1, 35.25)])
            for i in range(len(ids))
        ],
        crs="epsg:4326",
    )


def test_a_transect_the_config_does_not_hold_is_kept_with_a_nan_tide(
    stub_predictions, tmp_path
):
    observations = long_timeseries(
        ["t1", "t2", "t3"],
        ["2021-01-15"] * 3,
        cross_distance=[10.0, 20.0, 30.0],
    )
    # the config_gdf knows t1 and t2 but not t3
    result = tc._resolve_tides(
        observations,
        _one_transect_gdf(["t1", "t2"]),
        roi_id="roi_1",
        model_location=str(tmp_path),
    )

    assert len(result) == len(observations)
    assert sorted(result["transect_id"]) == ["t1", "t2", "t3"]
    assert result.loc[result["transect_id"] == "t3", "tide"].isna().all()


def test_two_identical_observations_both_survive(stub_predictions, tmp_path):
    """A raw frame can hold two rows that differ in nothing the merge keys see."""
    observations = long_timeseries(
        ["t1", "t1"], ["2021-01-15"] * 2, cross_distance=[10.0, 10.0]
    )

    result = tc._resolve_tides(
        observations, _one_transect_gdf(), roi_id="roi_1", model_location=str(tmp_path)
    )

    assert len(result) == 2
    assert result["tide"].tolist() == [1.5, 1.5]


def test_an_observation_with_no_cross_distance_is_kept(stub_predictions, tmp_path):
    """No cross_distance means no tide is predicted for that date, which used to take
    the whole row with it."""
    observations = long_timeseries(
        ["t1", "t1"],
        ["2021-01-15", "2021-02-15"],
        cross_distance=[10.0, np.nan],
    )

    result = tc._resolve_tides(
        observations, _one_transect_gdf(), roi_id="roi_1", model_location=str(tmp_path)
    )

    assert len(result) == 2
    assert result["tide"].isna().sum() == 1


# --------------------------------------------------------------------------
# a tides file replaces the model, so it must not require one
# --------------------------------------------------------------------------


def test_a_tides_file_does_not_need_the_model_downloaded(tmp_path, monkeypatch):
    """model defaults to 'FES2022', so gating the model check on it alone meant a
    caller who supplied their own tides still got "Tide model not found" from a run
    that was never going to open it."""
    monkeypatch.setattr(tc.core_utilities, "get_base_dir", lambda: str(tmp_path))
    looked_up = []
    monkeypatch.setattr(
        tc,
        "locate_tide_model",
        lambda *a, **k: looked_up.append(k) or (str(tmp_path), "global"),
    )
    monkeypatch.setattr(tc, "correct_tides", lambda *a, **k: pd.DataFrame())

    tides_file = tmp_path / "tides.csv"
    pd.DataFrame(
        {"dates": ["2021-01-15"], "transect_id": ["t1"], "tide": [1.2]}
    ).to_csv(tides_file, index=False)

    tc.correct_all_tides(
        ["roi_1"],
        "session",
        0.0,
        0.1,
        tides_file=str(tides_file),
        use_progress_bar=False,
    )

    assert looked_up == [], "the tide model must not be consulted for a tides file"


def test_without_a_tides_file_the_model_is_still_validated(tmp_path, monkeypatch):
    looked_up = []
    monkeypatch.setattr(
        tc,
        "locate_tide_model",
        lambda *a, **k: looked_up.append(k) or (str(tmp_path), "global"),
    )
    monkeypatch.setattr(tc, "correct_tides", lambda *a, **k: pd.DataFrame())

    tc.correct_all_tides(["roi_1"], "session", 0.0, 0.1, use_progress_bar=False)

    assert looked_up, "the tide model is the only source of tides here"


def test_an_roi_with_no_predictions_still_yields_a_float_tide_column(
    stub_predictions, tmp_path, monkeypatch
):
    """When nothing matched, the merge must still leave a numeric tide column.

    An empty prediction frame used to be built from column names alone, so every
    column was object dtype and the merged 'tide' came through as an object NaN,
    which then had to survive the correction arithmetic and a 'mean' pivot.
    """
    monkeypatch.setattr(
        tp,
        "model_tides",
        lambda *a, **k: pd.DataFrame(columns=["dates", "x", "y", "tide"]),
    )
    observations = long_timeseries(["t1"], ["2021-01-15"], cross_distance=10.0)

    # the config_gdf holds a transect the time series has never heard of
    result = tc._resolve_tides(
        observations,
        _one_transect_gdf(["somewhere_else"]),
        roi_id="roi_1",
        model_location=str(tmp_path),
    )

    assert len(result) == 1
    assert result["tide"].dtype == "float64"
    assert result["tide"].isna().all()


def _capture_correct_tides(monkeypatch, tmp_path, layout):
    """Run correct_all_tides with the disk reads stubbed, returning the kwargs."""
    captured = {}
    # stub the locate step, but let the real resolve_model_layout decide, so the
    # requested layout is genuinely carried through rather than handed back
    monkeypatch.setattr(
        tc,
        "locate_tide_model",
        lambda location="", model="fes2022", tide_model_layout="auto": (
            str(tmp_path),
            tc.resolve_model_layout(
                tmp_path, model, tide_model_layout=tide_model_layout
            ),
        ),
    )
    monkeypatch.setattr(
        tc,
        "correct_tides",
        lambda *a, **k: captured.update(k) or pd.DataFrame(),
    )

    tc.correct_all_tides(
        ["roi_1"],
        "session",
        0.0,
        0.1,
        use_progress_bar=False,
        tide_model_layout=layout,
    )
    return captured


@pytest.mark.parametrize(
    "requested, layout",
    [("clipped", "regions"), ("unclipped", "global")],
)
def test_the_resolved_layout_is_threaded_to_each_roi(
    tmp_path, monkeypatch, requested, layout
):
    """correct_all_tides resolves once and hands the answer down, so every ROI reads
    the same layout and the probe does not repeat."""
    captured = _capture_correct_tides(monkeypatch, tmp_path, requested)

    assert captured["tide_model_layout"] == layout


def test_the_layout_is_resolved_once_not_per_roi(tmp_path, monkeypatch):
    """A per-ROI probe let the region map and config['LAYOUT'] disagree, and cost a
    34-file glob per ROI."""
    calls = []
    monkeypatch.setattr(
        tc,
        "resolve_model_layout",
        lambda *a, **k: calls.append(k) or "global",
    )
    # let the real locate_tide_model run, with only its disk checks stubbed out
    monkeypatch.setattr(tc, "require_model_layout", lambda *a, **k: None)
    monkeypatch.setattr(tc, "require_tide_groups", lambda *a, **k: ())
    monkeypatch.setattr(tc, "correct_tides", lambda *a, **k: pd.DataFrame())

    tc.correct_all_tides(
        ["roi_1", "roi_2", "roi_3"], "session", 0.0, 0.1, use_progress_bar=False
    )

    # Two calls, not one per ROI: locate_tide_model probes the disk once, then
    # building the run's TideSource re-resolves the layout it was just handed,
    # which returns straight away without reading anything.
    assert len(calls) == 2
    probing = [call for call in calls if call["tide_model_layout"] == "auto"]
    assert len(probing) == 1


def test_the_model_root_override_reaches_the_lookup(tmp_path, monkeypatch):
    """A clipped model built somewhere other than CoastSeg/tide_model is only
    reachable if the path is actually plumbed through, not accepted and dropped."""
    seen = []
    monkeypatch.setattr(
        tc,
        "locate_tide_model",
        lambda *a, **k: seen.append((a, k)) or (str(tmp_path), "global"),
    )
    monkeypatch.setattr(tc, "correct_tides", lambda *a, **k: pd.DataFrame())

    tc.correct_all_tides(
        ["roi_1"],
        "session",
        0.0,
        0.1,
        use_progress_bar=False,
        tide_model_location=str(tmp_path / "elsewhere"),
    )

    args, kwargs = seen[0]
    assert args[0] == str(tmp_path / "elsewhere")
    assert kwargs["tide_model_layout"] == "auto"


def test_a_bad_layout_is_rejected_before_anything_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tc,
        "locate_tide_model",
        lambda *a, **k: pytest.fail("a typo must not reach the model lookup"),
    )

    with pytest.raises(ValueError, match="Unsupported tide model layout"):
        tc.correct_all_tides(
            ["roi_1"],
            "session",
            0.0,
            0.1,
            use_progress_bar=False,
            tide_model_layout="unclippped",
        )


def test_save_transect_settings_records_the_tide_model():
    """Which model and layout produced a session is not recoverable from its outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_file = os.path.join(tmpdir, "transects_settings.json")
        with open(settings_file, "w") as f:
            json.dump({"reference_elevation": 0, "beach_slope": 0}, f)

        save_transect_settings(
            tmpdir,
            1.23,
            4.56,
            tide_model="FES2022",
            tide_model_layout="clipped",
            tide_model_location=tmpdir,
        )

        with open(settings_file, "r") as f:
            settings = json.load(f)

        assert settings["tide_model"] == "FES2022"
        # the user-facing spelling, so it can be handed straight back to the API
        assert settings["tide_model_layout"] == "clipped"
        assert settings["tide_model_location"] == tmpdir


def _capture_predict_tides(monkeypatch):
    """Stub predict_tides, recording the region map path and layout it was handed."""
    captured = {}

    def fake(transects_gdf, timeseries_df, regions_path, config):
        captured["regions_path"] = regions_path
        captured["layout"] = config["LAYOUT"]
        return pd.DataFrame(columns=["dates", "x", "y", "tide", "transect_id"])

    monkeypatch.setattr(tc, "predict_tides", fake)
    return captured


def test_the_clipped_layout_finds_its_own_region_map(tmp_path, monkeypatch):
    """A caller who asks for the clipped layout but passes no region map used to reach
    load_regions_from_geojson("") inside predict_tides, whose DataSourceError says
    "No such file or directory" without naming a file."""
    captured = _capture_predict_tides(monkeypatch)

    tc._resolve_tides(
        _observations(),
        _one_transect_gdf(["t1"]),
        roi_id="roi_1",
        model_location=str(tmp_path),
        tide_model_layout="clipped",
    )

    assert captured["layout"] == "regions"
    assert captured["regions_path"].endswith("tide_regions_map.geojson")
    assert os.path.isfile(captured["regions_path"])


def test_auto_that_resolves_to_regions_still_finds_the_region_map(tmp_path, monkeypatch):
    """The hole an earlier version left open. Keying the lookup off the *requested*
    layout meant 'auto' on a regions-only install never resolved the map, and
    predict_tides read "" and reported "No such file or directory" with no path in it.
    It is now keyed off the config's LAYOUT, which is what predict_tides branches on."""
    captured = _capture_predict_tides(monkeypatch)
    monkeypatch.setattr(tc, "resolve_model_layout", lambda *a, **k: "regions")

    tc._resolve_tides(
        _observations(),
        _one_transect_gdf(["t1"]),
        roi_id="roi_1",
        model_location=str(tmp_path),
        tide_model_layout="auto",
    )

    assert captured["layout"] == "regions"
    assert captured["regions_path"].endswith("tide_regions_map.geojson")


@pytest.mark.parametrize("layout", ["unclipped", "auto"])
def test_only_the_clipped_layout_looks_for_the_region_map(tmp_path, monkeypatch, layout):
    """The un-clipped layout never reads it, and 'auto' would cost a probe per ROI."""
    captured = _capture_predict_tides(monkeypatch)
    monkeypatch.setattr(tc, "resolve_model_layout", lambda *a, **k: "global")
    monkeypatch.setattr(
        tc.file_utilities,
        "load_package_resource",
        lambda *a, **k: pytest.fail("the region map was loaded for a non-clipped run"),
    )

    tc._resolve_tides(
        _observations(),
        _one_transect_gdf(["t1"]),
        roi_id="roi_1",
        model_location=str(tmp_path),
        tide_model_layout=layout,
    )

    assert captured["regions_path"] == ""


def test_an_explicit_region_map_is_not_overridden(tmp_path, monkeypatch):
    captured = _capture_predict_tides(monkeypatch)

    tc._resolve_tides(
        _observations(),
        _one_transect_gdf(["t1"]),
        roi_id="roi_1",
        model_location=str(tmp_path),
        tide_regions_file="mine.geojson",
        tide_model_layout="clipped",
    )

    assert captured["regions_path"] == "mine.geojson"


def test_the_recorded_model_location_is_portable(monkeypatch, tmp_path):
    """Session folders get copied between machines, and an absolute path baked into one
    is dead everywhere else and carries the running user's home directory with it."""
    monkeypatch.setattr(tc.core_utilities, "get_base_dir", lambda: tmp_path)

    inside = tmp_path / "tide_model" / "fes2022b"
    assert tc.portable_model_location(str(inside)) == "tide_model/fes2022b"
    # a model outside the CoastSeg tree cannot be made relative, and is kept verbatim
    # rather than silently rewritten
    outside = os.path.join(os.path.abspath(os.sep), "elsewhere", "tide_model")
    assert tc.portable_model_location(outside) == pathlib.Path(outside).as_posix()
    assert tc.portable_model_location("") == ""


def _minimal_session(tmp_path, monkeypatch):
    """A session directory correct_tides can read, sandboxed under tmp_path."""
    monkeypatch.setattr(tc.core_utilities, "get_base_dir", lambda: tmp_path)
    roi_dir = tmp_path / "sessions" / "s" / "ID_roi"
    roi_dir.mkdir(parents=True)
    (roi_dir / "config.json").write_text("{}")
    long_timeseries(["t1"], ["2021-01-15"], cross_distance=10.0).to_csv(
        roi_dir / "raw_transect_time_series_merged.csv", index=False
    )
    monkeypatch.setattr(tc, "get_transects", lambda *a, **k: _one_transect_gdf(["t1"]))
    return roi_dir


def test_a_failed_correction_records_no_tide_model(tmp_path, monkeypatch):
    """Provenance written before the prediction claimed a model on sessions that then
    failed and wrote no tidally_corrected_* file at all."""
    roi_dir = _minimal_session(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("tide model exploded")

    monkeypatch.setattr(tc, "predict_tides", boom)

    with pytest.raises(RuntimeError, match="exploded"):
        tc.correct_tides(
            "roi", "s", 0.0, 0.02, model="FES2022",
            model_location=str(tmp_path), tide_model_layout="clipped",
            use_progress_bar=False,
        )

    settings = json.loads((roi_dir / "transects_settings.json").read_text())
    # the inputs are still recorded, as they always were
    assert settings["reference_elevation"] == 0.0
    assert not any(key.startswith("tide_model") for key in settings), settings


def test_a_successful_correction_records_the_tide_model(tmp_path, monkeypatch):
    roi_dir = _minimal_session(tmp_path, monkeypatch)
    monkeypatch.setattr(
        tc,
        "predict_tides",
        lambda *a, **k: pd.DataFrame(
            {"dates": pd.to_datetime(["2021-01-15"], utc=True), "x": [0.0],
             "y": [0.0], "tide": [0.5], "transect_id": ["t1"]}
        ),
    )

    tc.correct_tides(
        "roi", "s", 0.0, 0.02, model="FES2022",
        model_location=str(tmp_path / "tide_model"), tide_model_layout="clipped",
        use_progress_bar=False,
    )

    settings = json.loads((roi_dir / "transects_settings.json").read_text())
    assert settings["tide_model"] == "FES2022"
    assert settings["tide_model_layout"] == "clipped"
    assert settings["tide_model_location"] == "tide_model"

# --------------------------------------------------------------------------
# tide_source takes precedence, but not over the settings record
#
# correct_tides accepts both a TideSource and the five arguments one would be
# built from. The docstring says tide_source wins for *where the tides come
# from*, that four of the five still shape transects_settings.json, and that
# tide_regions_file alone becomes a complete no-op. Pin all three, or the
# documentation drifts away from the code.
# --------------------------------------------------------------------------


def _counting_source(tide=0.5):
    """A TideSource that records how often it ran and fills a constant tide."""
    calls = []

    def resolve(timeseries, transects_gdf):
        calls.append(len(timeseries))
        return timeseries.assign(tide=tide)

    return tc.TideSource("Stub tides", resolve), calls


def test_tide_source_wins_over_the_arguments_it_would_be_built_from(
    tmp_path, monkeypatch
):
    """The model is never consulted when a source is supplied, even though model=
    and model_location= name one."""
    _minimal_session(tmp_path, monkeypatch)

    def explode(*args, **kwargs):
        raise AssertionError("the tide model was consulted despite tide_source")

    monkeypatch.setattr(tc, "predict_tides", explode)
    monkeypatch.setattr(tc, "setup_tide_model_config", explode)
    source, calls = _counting_source()

    corrected = tc.correct_tides(
        "roi", "s", 0.0, 0.02,
        model="FES2022",
        model_location=str(tmp_path / "nowhere"),
        tide_model_layout="clipped",
        tide_source=source,
        use_progress_bar=False,
    )

    assert calls == [1]
    assert corrected["tide"].tolist() == [0.5]


def test_a_region_map_passed_beside_a_tide_source_is_a_no_op(tmp_path, monkeypatch):
    """tide_regions_file is the one argument with no other use, so it does nothing
    at all here. Documented so nobody expects a custom region map to be honoured."""
    _minimal_session(tmp_path, monkeypatch)
    source, calls = _counting_source()

    tc.correct_tides(
        "roi", "s", 0.0, 0.02,
        model="FES2022",
        model_location=str(tmp_path),
        tide_regions_file=str(tmp_path / "does_not_exist.geojson"),
        tide_source=source,
        use_progress_bar=False,
    )

    # a missing region map would have raised had anything tried to read it
    assert calls == [1]


def test_the_provenance_arguments_still_reach_the_settings_with_a_tide_source(
    tmp_path, monkeypatch
):
    """model, model_location and tide_model_layout keep shaping the settings record
    even though they no longer choose the tides."""
    roi_dir = _minimal_session(tmp_path, monkeypatch)
    source, _ = _counting_source()

    tc.correct_tides(
        "roi", "s", 0.0, 0.02,
        model="FES2014",
        model_location=str(tmp_path / "tide_model"),
        tide_model_layout="clipped",
        tide_source=source,
        use_progress_bar=False,
    )

    settings = json.loads((roi_dir / "transects_settings.json").read_text())
    assert settings["tide_model"] == "FES2014"
    assert settings["tide_model_layout"] == "clipped"
    assert settings["tide_model_location"] == "tide_model"


def test_a_tides_file_argument_still_suppresses_the_model_record(tmp_path, monkeypatch):
    """tides_file decides *whether* a model is recorded, and keeps doing so."""
    roi_dir = _minimal_session(tmp_path, monkeypatch)
    source, _ = _counting_source()

    tc.correct_tides(
        "roi", "s", 0.0, 0.02,
        model="FES2022",
        tides_file="anything.csv",
        model_location=str(tmp_path / "tide_model"),
        tide_source=source,
        use_progress_bar=False,
    )

    settings = json.loads((roi_dir / "transects_settings.json").read_text())
    assert not any(key.startswith("tide_model") for key in settings), settings
