from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "segmentation_workflow"
    / "run_zoo_segmentation_models.py"
)


def _load_standalone_module(monkeypatch) -> object:
    fake_logger = types.SimpleNamespace(setLevel=lambda *_args, **_kwargs: None)
    fake_tf = types.SimpleNamespace(
        get_logger=lambda: fake_logger,
        config=types.SimpleNamespace(
            set_visible_devices=lambda *_args, **_kwargs: None
        ),
    )

    fake_transformers = types.SimpleNamespace(TFSegformerForSemanticSegmentation=object)

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    module_name = "test_segmentation_workflow_model_info_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_write_model_info_json_contains_expected_fields(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_standalone_module(monkeypatch)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "model"
    input_dir.mkdir()
    output_dir.mkdir()
    model_dir.mkdir()

    config = module.SegmentationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=str(model_dir),
    )
    model_spec = module.ModelSpec(
        model=None,
        model_type="segformer",
        target_size=(512, 512),
        nclasses=4,
        class_mapping={0: "water", 1: "whitewater", 2: "sediment", 3: "other"},
    )
    backend = module.InferenceBackend(models=(model_spec,))

    model_info_path = module._write_model_info_json(config, backend)

    assert model_info_path.exists()
    payload = json.loads(model_info_path.read_text(encoding="utf-8"))

    assert payload["model_directory"] == str(model_dir.resolve())
    assert payload["input_directory"] == str(input_dir)
    assert payload["class_mapping"] == {
        "0": "water",
        "1": "whitewater",
        "2": "sediment",
        "3": "other",
    }
    assert payload["water_class_indices"] == [0, 1]
    assert payload["water_classes"] == ["water", "whitewater"]


def test_write_model_settings_json_contains_expected_fields(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_standalone_module(monkeypatch)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "global_segformer_RGB_4class_14036903"
    input_dir.mkdir()
    output_dir.mkdir()
    model_dir.mkdir()

    config = module.SegmentationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=str(model_dir),
        implementation="best",
        use_gpu=False,
    )

    model_settings_path = module._write_model_settings_json(config)
    assert model_settings_path.exists()

    payload = json.loads(model_settings_path.read_text(encoding="utf-8"))
    assert payload == {
        "use_GPU": "0",
        "implementation": "BEST",
        "model_type": "global_segformer_RGB_4class_14036903",
        "img_type": "RGB",
    }
