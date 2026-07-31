"""Tests for the warnings CoastSeg raises when SAR shorelines are not model-segmented.

coastsat never raises when the SAR segmentation model is unusable -- ``load_sar_segmenter``
returns None and every scene is thresholded with the legacy Otsu method instead. The run
still produces shorelines, so nothing looks wrong, and coastsat records the fallback only
in its own log inside the session folder. These tests pin the two places CoastSeg surfaces
it to the user:

1. a pre-flight check, before extraction, for the whole-run fallback, and
2. a post-run tally of ``output['segmentation_method']``, which also catches the per-scene
   fallbacks the pre-flight check cannot predict.

Nothing here loads the real ~130 MB ONNX model: the segmenter is stubbed.
"""

import numpy as np
import pytest
from coastsat import SDS_shoreline, SDS_tools

from coastseg import extracted_shoreline as es

S1_METADATA = {"S1": {"dates": ["2023-05-06-02-07-48"]}}


@pytest.fixture
def segmenter(monkeypatch):
    """Control what coastsat's load_sar_segmenter returns without touching the model."""

    def set_result(result):
        monkeypatch.setattr(
            SDS_shoreline, "load_sar_segmenter", lambda *a, **k: result
        )

    return set_result


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------


def test_an_unusable_model_warns_that_the_whole_run_falls_back(
    caplog, capsys, segmenter
):
    """The silent degrade is the whole point: a user who is not told keeps the worse
    shorelines and never knows the model was skipped."""
    segmenter(None)

    with caplog.at_level("WARNING"):
        fell_back = es.warn_if_sar_falls_back_to_otsu({}, S1_METADATA)

    assert fell_back is True
    assert "falls back to the legacy Otsu threshold" in caplog.text
    # printed too -- a notebook user does not read the logger
    assert "WARNING" in capsys.readouterr().out


def test_a_loadable_model_says_nothing(caplog, segmenter):
    segmenter(("session", {"channel_order": ["VV", "VH", "VV-VH"]}))

    with caplog.at_level("WARNING"):
        assert es.warn_if_sar_falls_back_to_otsu({}, S1_METADATA) is False

    assert caplog.text == ""


def test_asking_for_otsu_is_not_a_warning(caplog, capsys, segmenter):
    """Otsu chosen deliberately is a valid configuration, not a degraded run. Warning
    about it would train users to ignore the warning that matters."""
    segmenter(None)

    with caplog.at_level("INFO"):
        assert (
            es.warn_if_sar_falls_back_to_otsu(
                {"sar_segmentation": "otsu"}, S1_METADATA
            )
            is False
        )

    assert "WARNING" not in capsys.readouterr().out
    assert "thresholded rather than segmented" in caplog.text


@pytest.mark.parametrize("metadata", [{}, {"S2": {"dates": ["2023-05-06"]}}])
def test_an_optical_only_run_never_loads_the_sar_model(caplog, monkeypatch, metadata):
    """No S1 scenes means the SAR model is irrelevant, and loading a ~130 MB model to
    discover that would cost every optical run real time."""

    def explode(*args, **kwargs):
        raise AssertionError("the SAR model must not be loaded for an optical run")

    monkeypatch.setattr(SDS_shoreline, "load_sar_segmenter", explode)

    with caplog.at_level("WARNING"):
        assert es.warn_if_sar_falls_back_to_otsu({}, metadata) is False

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# Post-run tally
# ---------------------------------------------------------------------------


def test_a_per_scene_fallback_is_reported_after_the_run(caplog, capsys):
    """A scene coastsat could not segment drops to Otsu on its own, which the pre-flight
    check cannot predict."""
    output = {
        "segmentation_method": [
            SDS_tools.SEGMENTATION_SAR_MODEL,
            SDS_tools.SEGMENTATION_SAR_OTSU,
            SDS_tools.SEGMENTATION_SAR_MODEL,
        ]
    }

    with caplog.at_level("WARNING"):
        counts = es.report_sar_segmentation_methods(output)

    assert counts == {
        SDS_tools.SEGMENTATION_SAR_MODEL: 2,
        SDS_tools.SEGMENTATION_SAR_OTSU: 1,
    }
    assert "1 of 3" in caplog.text
    assert "WARNING" in capsys.readouterr().out


def test_a_fully_model_segmented_run_does_not_warn(caplog, capsys):
    output = {
        "segmentation_method": [SDS_tools.SEGMENTATION_SAR_MODEL] * 3
    }

    with caplog.at_level("INFO"):
        counts = es.report_sar_segmentation_methods(output)

    assert counts[SDS_tools.SEGMENTATION_SAR_OTSU] == 0
    assert "WARNING" not in capsys.readouterr().out
    assert "All 3 Sentinel-1 shorelines" in caplog.text


def test_an_optical_only_output_is_not_reported_on(caplog, capsys):
    """Optical shorelines carry 'mndwi' and must not be counted as SAR."""
    output = {"segmentation_method": [SDS_tools.SEGMENTATION_MNDWI] * 4}

    with caplog.at_level("INFO"):
        counts = es.report_sar_segmentation_methods(output)

    assert counts == {
        SDS_tools.SEGMENTATION_SAR_MODEL: 0,
        SDS_tools.SEGMENTATION_SAR_OTSU: 0,
    }
    assert capsys.readouterr().out == ""


def test_output_without_the_column_is_handled():
    """CoastSeg's own zoo extraction records no SAR methods; this must not raise."""
    assert es.report_sar_segmentation_methods({}) == {
        SDS_tools.SEGMENTATION_SAR_MODEL: 0,
        SDS_tools.SEGMENTATION_SAR_OTSU: 0,
    }
