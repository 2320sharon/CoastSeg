"""Unit tests for the Sentinel-1 ONNX segmentation preprocessing contract.

Getting any preprocessing step wrong does not crash -- it silently degrades or destroys
accuracy -- so each step of the documented pipeline is pinned here.
"""

import os

import numpy as np
import pytest

from coastseg import sar_model

SAR_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "SAR_segmentation_model",
)

requires_model = pytest.mark.skipif(
    not os.path.isdir(SAR_MODEL_DIR) or not any(
        name.endswith(".onnx") for name in os.listdir(SAR_MODEL_DIR)
    ),
    reason="the SAR ONNX model is not present in models/SAR_segmentation_model",
)


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------


@requires_model
def test_load_spec_reads_metadata_props():
    """Every constant comes from the model file, not from hardcoded values."""
    spec = sar_model.load_sar_model_spec(SAR_MODEL_DIR)

    assert spec.channel_order == ("VV", "VH", "VV-VH")
    assert np.allclose(spec.mean, [-12.59, -20.26, 10.5465])
    assert np.allclose(spec.std, [5.26, 5.91, 7.6855])
    assert spec.output_stride == 32
    assert spec.nodata == 255
    assert spec.classes == {0: "land", 1: "water"}
    assert spec.input_name == "image"
    assert spec.prob_output == "water_prob"
    assert spec.input_scale == "dB"
    assert spec.speckle_filter["type"] == "none"


# The exported model exposes both of these, so which one carries P(water) is a real
# choice, not a hypothetical one. Thresholding logits at 0.5 does not raise -- it just
# maps the wrong shoreline.
BOTH_OUTPUTS = ["logits", "water_prob"]


def test_prob_output_prefers_the_named_water_probability():
    assert (
        sar_model._resolve_prob_output(BOTH_OUTPUTS, BOTH_OUTPUTS, "m.onnx")
        == "water_prob"
    )


def test_prob_output_is_not_guessed_by_position():
    """'water_prob' wins wherever it sits, so export order cannot change the result."""
    reordered = ["water_prob", "logits"]

    assert (
        sar_model._resolve_prob_output(reordered, reordered, "m.onnx") == "water_prob"
    )


def test_ambiguous_outputs_raise_instead_of_picking_one():
    """Falling back to the last output would silently threshold logits at 0.5."""
    names = ["logits", "features"]

    with pytest.raises(sar_model.SarModelError, match="water_prob"):
        sar_model._resolve_prob_output(names, names, "retrained.onnx")


def test_a_single_output_is_used_but_says_so(caplog):
    """One output is unambiguous; the run continues, but not silently."""
    with caplog.at_level("WARNING"):
        chosen = sar_model._resolve_prob_output(["prob"], ["prob"], "m.onnx")

    assert chosen == "prob"
    assert any("water_prob" in record.message for record in caplog.records)


def test_metadata_naming_an_output_the_graph_lacks_raises():
    """Otherwise session.run fails later with an opaque onnxruntime error."""
    with pytest.raises(sar_model.SarModelError, match="graph exposes"):
        sar_model._resolve_prob_output(
            ["water_prob"], ["logits", "probability"], "m.onnx"
        )


def test_water_class_indices_is_derived_from_class_names(sar_model_spec_stub):
    # This model's water class is 1, the inverse of the SegFormer convention.
    assert sar_model_spec_stub.water_class_indices == [1]


def test_required_polarizations_expands_difference_channels(sar_model_spec_stub):
    """A VV-VH channel contributes both operands, not a literal 'VV-VH' band."""
    assert sar_model_spec_stub.required_polarizations == ("VV", "VH")


def test_is_sar_model_directory(tmp_path, sar_scene_tifs):
    assert not sar_model.is_sar_model_directory("")
    assert not sar_model.is_sar_model_directory(str(tmp_path))
    (tmp_path / "some_model.onnx").write_bytes(b"")
    assert sar_model.is_sar_model_directory(str(tmp_path))


def test_the_designated_directory_routes_as_sar_while_still_empty(tmp_path):
    """The model is downloaded on first use, so the designated directory is empty (or
    absent) until then. If routing depended on the file being there, the first SAR run
    on a fresh clone would be sent down the TensorFlow path instead."""
    designated = tmp_path / sar_model.DEFAULT_SAR_MODEL_DIRNAME

    assert sar_model.is_sar_model_directory(str(designated))  # does not exist yet
    designated.mkdir()
    assert sar_model.is_sar_model_directory(str(designated))  # exists but empty


def test_find_sar_onnx_file_missing_raises_when_download_is_off(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .onnx file"):
        sar_model.find_sar_onnx_file(str(tmp_path), download=False)


# ---------------------------------------------------------------------------
# Band stack (steps 1-7)
# ---------------------------------------------------------------------------


def test_difference_channel_uses_raw_db_before_any_fill(
    sar_scene_tifs, sar_model_spec_stub
):
    """VV-VH must be derived before invalid pixels are filled.

    Otherwise a fill value leaks into the subtraction and the third channel is wrong
    wherever the scene has nodata.
    """
    stack, valid, _, _ = sar_model.build_sar_stack(
        [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], sar_model_spec_stub
    )
    assert stack.shape == (3, 50, 70)
    assert np.allclose(stack[2], stack[0] - stack[1], equal_nan=True)


def test_validity_mask_is_the_intersection_of_both_polarizations(
    sar_scene_tifs, sar_model_spec_stub
):
    _, valid, _, _ = sar_model.build_sar_stack(
        [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], sar_model_spec_stub
    )
    height, width = sar_scene_tifs["shape"]
    nodata_rows = sar_scene_tifs["nodata_rows"]
    assert valid.shape == (height, width)
    # VV's last rows are nodata, so they must be invalid in the combined mask
    assert not valid[-nodata_rows:, :].any()
    assert valid[: height - nodata_rows, :].all()


def test_missing_polarization_raises(sar_site_vh_only, sar_model_spec_stub):
    with pytest.raises(sar_model.MissingPolarizationError, match="VV"):
        sar_model.build_sar_stack([sar_site_vh_only["vh"]], sar_model_spec_stub)


def test_grid_mismatch_reprojects_onto_the_reference_grid(
    sar_grid_mismatch_tifs, sar_model_spec_stub, caplog
):
    """A VH raster on a different grid is warped onto VV's grid, never the reverse."""
    with caplog.at_level("WARNING"):
        stack, valid, geotransform, _ = sar_model.build_sar_stack(
            [sar_grid_mismatch_tifs["vv"], sar_grid_mismatch_tifs["vh"]],
            sar_model_spec_stub,
        )

    assert stack.shape == (3,) + sar_grid_mismatch_tifs["shape"]
    assert valid.shape == sar_grid_mismatch_tifs["shape"]
    # georeferencing stays on the VV (reference) grid
    assert geotransform[1] == 10.0
    assert any("reprojecting" in record.message for record in caplog.records)


def test_unsupported_speckle_filter_raises(sar_scene_tifs, sar_model_spec_stub):
    """A model trained with a filter this module cannot apply must stop, not proceed."""
    import dataclasses

    spec = dataclasses.replace(
        sar_model_spec_stub, speckle_filter={"type": "lee_sigma"}
    )
    with pytest.raises(sar_model.UnsupportedSpeckleFilterError, match="lee_sigma"):
        sar_model.build_sar_stack(
            [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], spec
        )


# ---------------------------------------------------------------------------
# Fill / normalize / pad (steps 8-10)
# ---------------------------------------------------------------------------


def test_invalid_pixels_normalize_to_exactly_zero(sar_model_spec_stub):
    """Filling with the per-channel mean puts invalid pixels at a neutral 0."""
    spec = sar_model_spec_stub
    stack = np.zeros((3, 32, 32), dtype=np.float32)
    stack[:, :, :] = 999.0  # an extreme outlier everywhere
    valid = np.ones((32, 32), dtype=bool)
    valid[0, 0] = False

    tensor, _ = sar_model.preprocess(stack, valid, spec)
    assert np.allclose(tensor[0, :, 0, 0], 0.0)


def test_normalization_is_per_channel(sar_model_spec_stub):
    spec = sar_model_spec_stub
    stack = np.stack(
        [
            np.full((32, 32), -10.0, dtype=np.float32),
            np.full((32, 32), -18.0, dtype=np.float32),
            np.full((32, 32), 8.0, dtype=np.float32),
        ]
    )
    valid = np.ones((32, 32), dtype=bool)

    tensor, _ = sar_model.preprocess(stack, valid, spec)
    expected = (np.array([-10.0, -18.0, 8.0], dtype=np.float32) - spec.mean) / spec.std
    assert np.allclose(tensor[0, :, 5, 5], expected)


def test_pads_bottom_right_to_a_multiple_of_the_output_stride(sar_model_spec_stub):
    """50x70 becomes 64x96, padded with zeros on the bottom and right only."""
    spec = sar_model_spec_stub
    stack = np.ones((3, 50, 70), dtype=np.float32)
    valid = np.ones((50, 70), dtype=bool)

    tensor, original_shape = sar_model.preprocess(stack, valid, spec)

    assert tensor.shape == (1, 3, 64, 96)
    assert original_shape == (50, 70)
    assert np.all(tensor[0, :, 50:, :] == 0.0)
    assert np.all(tensor[0, :, :, 70:] == 0.0)
    # nothing was padded on the top or left
    assert not np.all(tensor[0, :, 0, 0] == 0.0)


def test_already_aligned_shape_is_not_padded(sar_model_spec_stub):
    stack = np.ones((3, 64, 32), dtype=np.float32)
    valid = np.ones((64, 32), dtype=bool)
    tensor, _ = sar_model.preprocess(stack, valid, sar_model_spec_stub)
    assert tensor.shape == (1, 3, 64, 32)


def test_oversized_scene_raises_instead_of_exhausting_memory(sar_model_spec_stub, monkeypatch):
    monkeypatch.setattr(sar_model, "MAX_INPUT_PIXELS", 100)
    stack = np.ones((3, 64, 64), dtype=np.float32)
    valid = np.ones((64, 64), dtype=bool)
    with pytest.raises(sar_model.SarModelError, match="pixel limit|above the"):
        sar_model.preprocess(stack, valid, sar_model_spec_stub)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def test_postprocess_crops_padding_and_never_resizes(sar_model_spec_stub):
    probability = np.zeros((1, 1, 64, 96), dtype=np.float32)
    valid = np.ones((50, 70), dtype=bool)

    label, cropped = sar_model.postprocess(probability, (50, 70), valid, sar_model_spec_stub)

    assert label.shape == (50, 70)
    assert cropped.shape == (50, 70)


def test_threshold_boundary_is_inclusive(sar_model_spec_stub):
    probability = np.array([[0.5, 0.4999, 0.9]], dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)

    label, _ = sar_model.postprocess(probability, (1, 3), valid, sar_model_spec_stub)

    assert label.tolist() == [[1, 0, 1]]


def test_nodata_is_stamped_onto_invalid_pixels(sar_model_spec_stub):
    probability = np.full((1, 1, 32, 32), 0.9, dtype=np.float32)
    valid = np.ones((32, 32), dtype=bool)
    valid[0, :] = False

    label, cropped = sar_model.postprocess(
        probability, (32, 32), valid, sar_model_spec_stub
    )

    assert np.all(label[0, :] == 255)
    assert np.all(label[1:, :] == 1)
    assert np.all(np.isnan(cropped[0, :]))
    assert not np.any(np.isnan(cropped[1:, :]))


def test_labels_follow_the_model_class_indices_not_a_fixed_water_value(
    sar_model_spec_stub,
):
    """A model free to declare {0: "water", 1: "land"} must not come out inverted.

    Extraction decodes ``grey_label`` by testing equality against
    ``model_info.water_class_indices``, which comes from these same class names. Writing
    a fixed 1 for water would silently swap land and sea on such a model -- a plausible
    looking shoreline on the wrong side of the beach.
    """
    import dataclasses

    probability = np.array([[0.9, 0.1]], dtype=np.float32)
    valid = np.ones((1, 2), dtype=bool)

    reversed_spec = dataclasses.replace(
        sar_model_spec_stub, classes={0: "water", 1: "land"}
    )
    label, _ = sar_model.postprocess(probability, (1, 2), valid, reversed_spec)

    assert reversed_spec.water_class_indices == [0]
    assert label.tolist() == [[0, 1]]  # water pixel carries the water class index


def test_labels_still_use_1_for_water_on_the_conventional_mapping(sar_model_spec_stub):
    """The shipped {0: land, 1: water} model keeps producing exactly what it did."""
    probability = np.array([[0.9, 0.1]], dtype=np.float32)
    valid = np.ones((1, 2), dtype=bool)

    label, _ = sar_model.postprocess(probability, (1, 2), valid, sar_model_spec_stub)

    assert sar_model_spec_stub.water_class_indices == [1]
    assert label.tolist() == [[1, 0]]


# ---------------------------------------------------------------------------
# Scene / site helpers
# ---------------------------------------------------------------------------


def test_output_stem_matches_the_tensorflow_zoo_convention():
    """The stem must stay {date}_RGB_S1 so npz lookups keep matching."""
    assert (
        sar_model._output_stem("2023-05-01-04-26-29_S1_site_VV.tif")
        == "2023-05-01-04-26-29_RGB_S1"
    )


def test_output_stem_keeps_duplicate_suffix():
    """Duplicate acquisitions must not overwrite each other's artifacts."""
    assert (
        sar_model._output_stem("2023-05-01-04-26-29_S1_site_VV_dup0.tif")
        == "2023-05-01-04-26-29_RGB_S1_dup0"
    )


def test_collect_sar_scenes_pairs_polarizations(sar_scene_tifs):
    scenes = sar_model.collect_sar_scenes(sar_scene_tifs["site"])
    assert len(scenes) == 1
    (paths,) = scenes.values()
    assert set(paths) == {"VV", "VH"}


def test_collect_sar_scenes_on_a_legacy_site(sar_site_vh_only):
    scenes = sar_model.collect_sar_scenes(sar_site_vh_only["site"])
    (paths,) = scenes.values()
    assert set(paths) == {"VH"}


def test_polarization_token_is_not_confused_by_a_sitename():
    """A sitename containing 'VH' must not be read as a polarization token."""
    assert sar_model.get_polarization_from_filename("2023_S1_myVHsite_VV.tif") == "VV"


# ---------------------------------------------------------------------------
# Nodata helper
# ---------------------------------------------------------------------------


def test_sar_nodata_mask_reports_the_empty_part_of_the_swath(sar_scene_tifs):
    mask = sar_model.sar_nodata_mask([sar_scene_tifs["vv"], sar_scene_tifs["vh"]])
    height, width = sar_scene_tifs["shape"]
    assert mask.shape == (height, width)
    assert int(mask.sum()) == sar_scene_tifs["nodata_rows"] * width


def test_sar_nodata_mask_returns_none_when_unreadable(tmp_path):
    assert sar_model.sar_nodata_mask([str(tmp_path / "missing.tif")]) is None


# ---------------------------------------------------------------------------
# Real inference
# ---------------------------------------------------------------------------


@requires_model
def test_real_model_separates_water_from_land(sar_scene_tifs):
    """Low-backscatter pixels come back as water and high-backscatter ones as land."""
    spec = sar_model.load_sar_model_spec(SAR_MODEL_DIR)
    prediction = sar_model.segment_scene(
        [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], spec
    )

    height, width = sar_scene_tifs["shape"]
    valid_rows = height - sar_scene_tifs["nodata_rows"]

    assert prediction.label.shape == (height, width)
    assert set(np.unique(prediction.label).tolist()) <= {0, 1, 255}
    assert (prediction.label[:valid_rows, : width // 2] == 1).mean() > 0.9
    assert (prediction.label[:valid_rows, width // 2 :] == 1).mean() < 0.1
