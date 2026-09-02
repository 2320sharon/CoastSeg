"""Tests for how the zoo SAR workflow obtains its ONNX model.

The model is not shipped with CoastSeg -- it is ~130 MB and ``models/**/*.onnx`` is
gitignored, so a fresh clone has none. ``find_sar_onnx_file`` downloads it from Hugging
Face on first use, deferring to ``coastsat.SDS_sar_model`` so both SAR workflows share
one sha256-pinned pooch cache instead of keeping a copy each.

Nothing here touches the network: coastsat's resolver and fetcher are both stubbed. A
test that reached Hugging Face would download 130 MB per run.
"""

import os

import pytest
from coastsat import SDS_sar_model

from coastseg import sar_model


@pytest.fixture(autouse=True)
def no_real_downloads(monkeypatch, tmp_path):
    """Make the default model look absent and the download explode.

    Autouse so a test that accidentally reaches the download path fails loudly instead
    of silently pulling 130 MB. Tests that mean to download override these.
    """
    monkeypatch.setattr(
        SDS_sar_model, "get_sar_model_path", lambda *a, **k: str(tmp_path / "absent.onnx")
    )

    def explode():
        raise AssertionError("the test tried to download the real SAR model")

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", explode)


@pytest.fixture
def fake_download(monkeypatch, tmp_path):
    """Stand in for the Hugging Face fetch, recording that it was called."""
    calls = []
    cached = tmp_path / "cache" / "SAR_3_band_model.onnx"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"onnx")

    def fetch():
        calls.append(True)
        return str(cached)

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", fetch)
    return {"calls": calls, "path": str(cached)}


# ---------------------------------------------------------------------------
# A local copy always wins
# ---------------------------------------------------------------------------


def test_a_local_model_is_used_without_downloading(tmp_path, fake_download):
    """A retrained or manually placed model must never be silently replaced by the
    default one, and an offline machine that already has a model must keep working."""
    local = tmp_path / "SAR_segmentation_model"
    local.mkdir()
    (local / "my_retrained.onnx").write_bytes(b"local")

    resolved = sar_model.find_sar_onnx_file(str(local))

    assert resolved == str((local / "my_retrained.onnx").resolve())
    assert fake_download["calls"] == []


def test_an_already_cached_default_is_not_re_downloaded(monkeypatch, tmp_path):
    """The second run must not re-fetch: coastsat's resolver already points at the
    cached copy, whichever workflow put it there."""
    cached = tmp_path / "SAR_3_band_model.onnx"
    cached.write_bytes(b"onnx")
    monkeypatch.setattr(SDS_sar_model, "get_sar_model_path", lambda *a, **k: str(cached))

    # fetch_default_sar_model is still the exploding stub from the autouse fixture
    resolved = sar_model.find_sar_onnx_file(str(tmp_path / "empty_dir"))

    assert resolved == os.path.abspath(str(cached))


# ---------------------------------------------------------------------------
# Downloading when there is nothing on disk
# ---------------------------------------------------------------------------


def test_an_empty_model_directory_downloads_the_default(tmp_path, fake_download):
    empty = tmp_path / sar_model.DEFAULT_SAR_MODEL_DIRNAME
    empty.mkdir()

    resolved = sar_model.find_sar_onnx_file(str(empty))

    assert resolved == fake_download["path"]
    assert fake_download["calls"] == [True]


def test_a_missing_model_directory_downloads_the_default(tmp_path, fake_download):
    """A fresh clone has no models/SAR_segmentation_model directory at all."""
    resolved = sar_model.find_sar_onnx_file(
        str(tmp_path / "never_created" / sar_model.DEFAULT_SAR_MODEL_DIRNAME)
    )

    assert resolved == fake_download["path"]
    assert fake_download["calls"] == [True]


def test_download_is_skippable(tmp_path):
    """download=False keeps the old fail-fast behaviour for callers that want it."""
    with pytest.raises(FileNotFoundError, match="No .onnx file"):
        sar_model.find_sar_onnx_file(str(tmp_path), download=False)


# ---------------------------------------------------------------------------
# Failure is actionable
# ---------------------------------------------------------------------------


def test_a_failed_download_explains_how_to_do_it_by_hand(monkeypatch, tmp_path):
    """Offline, behind a proxy, or Hugging Face down: the user needs the URL and the
    destination, not just 'download failed'."""

    def fail():
        raise SDS_sar_model.SarModelUnavailable("no internet")

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", fail)

    with pytest.raises(sar_model.SarModelError) as error:
        sar_model.find_sar_onnx_file(str(tmp_path / "empty"))

    message = str(error.value)
    assert "no internet" in message
    assert "huggingface.co" in message
    assert sar_model.DEFAULT_SAR_MODEL_DIRNAME in message


def test_the_download_failure_is_a_coastseg_error(monkeypatch, tmp_path):
    """Callers catch coastseg.sar_model.SarModelError; coastsat's exception type must
    not leak out of this module."""

    def fail():
        raise SDS_sar_model.SarModelUnavailable("nope")

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", fail)

    with pytest.raises(sar_model.SarModelError):
        sar_model.find_sar_onnx_file(str(tmp_path / "empty"))
