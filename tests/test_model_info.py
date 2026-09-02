from __future__ import annotations

import json
from pathlib import Path

import pytest

from coastseg.model_info import DEFAULT_CLASS_MAPPING, ModelInfo


def test_model_info_loads_from_model_info_json(tmp_path: Path) -> None:
    payload = {
        "model_directory": str(tmp_path),
        "input_directory": str(tmp_path / "input"),
        "class_mapping": {
            "0": "water",
            "1": "whitewater",
            "2": "sediment",
            "3": "other",
        },
        "water_class_indices": [0, 1],
        "water_classes": ["water", "whitewater"],
    }
    (tmp_path / "model_info.json").write_text(json.dumps(payload), encoding="utf-8")

    model_info = ModelInfo(model_directory=str(tmp_path))
    model_info.load_from_model_info_file(
        model_info_path=str(tmp_path / "model_info.json")
    )

    assert model_info.class_mapping == {
        0: "water",
        1: "whitewater",
        2: "sediment",
        3: "other",
    }
    assert model_info.water_class_indices == [0, 1]
    assert model_info.input_directory == str(tmp_path / "input")


def test_model_info_accepts_direct_model_info_path(tmp_path: Path) -> None:
    model_info_json = tmp_path / "model_info.json"
    payload = {
        "class_mapping": {
            "0": "water",
            "1": "whitewater",
        },
        "water_class_indices": [0, 1],
    }
    model_info_json.write_text(json.dumps(payload), encoding="utf-8")

    model_info = ModelInfo(model_info_path=str(model_info_json))
    model_info.load_from_model_info_file()

    assert model_info.model_info_path == str(model_info_json.resolve())
    assert model_info.model_directory == str(tmp_path.resolve())
    assert model_info.class_mapping == {0: "water", 1: "whitewater"}
    assert model_info.water_class_indices == [0, 1]


def test_model_info_accepts_model_directory_as_model_info_file(tmp_path: Path) -> None:
    model_info_json = tmp_path / "model_info.json"
    payload = {
        "class_mapping": {
            "0": "water",
            "1": "whitewater",
            "2": "sediment",
        },
        "water_classes": ["water"],
    }
    model_info_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ModelInfo(model_directory=str(model_info_json))


def test_model_info_resolves_water_indices_from_names(tmp_path: Path) -> None:
    payload = {
        "class_mapping": {
            "0": "water",
            "1": "whitewater",
            "2": "sediment",
        },
        "water_classes": ["whitewater"],
    }
    (tmp_path / "model_info.json").write_text(json.dumps(payload), encoding="utf-8")

    model_info = ModelInfo(model_directory=str(tmp_path))
    model_info.load_from_model_info_file(
        model_info_path=str(tmp_path / "model_info.json")
    )

    assert model_info.class_mapping[1] == "whitewater"
    assert model_info.water_class_indices == [1]


def test_model_info_defaults_without_modelcard_or_model_info(tmp_path: Path) -> None:
    model_info = ModelInfo(model_directory=str(tmp_path))
    assert model_info.class_mapping == DEFAULT_CLASS_MAPPING
    assert model_info.water_class_indices == [0, 1]


def test_model_info_can_load_with_stored_model_info_path_only(tmp_path: Path) -> None:
    model_info_json = tmp_path / "model_info.json"
    model_info_json.write_text(
        json.dumps({"class_mapping": {"0": "water"}}),
        encoding="utf-8",
    )

    model_info = ModelInfo(model_info_path=str(model_info_json))
    model_info.load_from_model_info_file()

    assert model_info.class_mapping == {0: "water"}


def test_model_info_load_from_model_info_file_raises_if_missing(tmp_path: Path) -> None:
    model_info = ModelInfo(model_directory=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        model_info.load_from_model_info_file(
            model_info_path=str(tmp_path / "model_info.json")
        )
