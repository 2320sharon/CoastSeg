import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
import os

from coastseg import common, file_utilities


DEFAULT_CLASS_MAPPING: Dict[int, str] = {
    0: "water",
    1: "whitewater",
    2: "sediment",
    3: "other",
}


@dataclass
class ModelInfo:
    """
    Stores and manages metadata for a segmentation model, including the class mapping and
    identification of water-related classes.

    This class resolves the model's directory from a name or a provided path, loads a class
    mapping from a model card file (typically in JSON format), and identifies indices
    corresponding to water-related classes (e.g., "water", "whitewater").

    Default class mapping is provided if no mapping is found in the modelcard file.
    The default mapping includes:
    - 0: "water"
    - 1: "whitewater"
    - 2: "sediment"
    - 3: "other"
    The `water_class_indices` are automatically computed based on the class names "water" and "whitewater".

    -----------------------
    Supported Initialization
    -----------------------

    1. From a model directory:
        Provide `model_directory`, optionally with `model_name`.
        Call `load()` to load the class mapping from a `modelcard.json` file in the directory.

        ⚠ The `model_directory` must contain a file matching the pattern 'modelcard.json'.
        ⚠ If no class mapping is found in the modelcard file, a default mapping is used.


        Example:
            model_info = ModelInfo(
                model_directory="models/my_model",
                model_name="my_model"
            )
            model_info.load()
            print(model_info.class_mapping)

    2. From a model name only:
        Provide `model_name`. The directory will be resolved using
        `common.get_downloaded_models_dir() / model_name`.

        Example:
            model_info = ModelInfo(model_name="AK_segformer")
            model_info.load()

    3. From a direct class mapping (no file access):
        Provide a `class_mapping` dictionary directly.
        Call `load()` to compute water-related indices based on this mapping.

        Example:
            model_info = ModelInfo(class_mapping={0: "water", 1: "land"})
            model_info.load()

    4. From a custom extractor function:
        Pass a function to `load(extractor=...)` that extracts a class mapping from a
        modelcard JSON dictionary.

        Example:
            def my_extractor(data):
                return {int(k): v for k, v in data["custom_labels"].items()}

            model_info = ModelInfo(model_directory="models/custom")
            model_info.load(extractor=my_extractor)

    Attributes:
        model_name (Optional[str]): Name of the model, used to locate the model directory if a path is not explicitly provided.
        model_directory (Optional[str]): Absolute path to the model directory. If not provided, it will be resolved using model_name.
        class_mapping (Dict[int, str]): Mapping from class indices to class labels. Loaded from a model card or provided directly.
        water_class_indices (List[int]): List of class indices corresponding to water-related classes.

    Methods:
        load(extractor: Optional[Callable[[dict], Dict[int, str]]] = None):
            Loads the class mapping and identifies water class indices. An optional extractor can be used to customize parsing of the model card.
    """

    model_name: Optional[str] = None
    model_directory: Optional[str] = None
    model_info_path: Optional[str] = None
    input_directory: Optional[str] = None
    class_mapping: Dict[int, str] = field(default_factory=dict)
    water_class_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initializes derived fields after dataclass construction.

        Resolves the model directory and loads class metadata.
        """
        self._resolve_model_info_path()
        # loads the model directory from the provided name or path, validating existence
        self.model_directory = self._resolve_directory()
        self.load()

    def _resolve_model_info_path(self) -> None:
        """Resolve and validate an explicit model_info.json path when provided.

        Raises:
            FileNotFoundError: If an explicitly provided model_info.json path does not exist.
        """
        if not self.model_info_path:
            return

        candidate = os.path.abspath(self.model_info_path)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"model_info.json file not found: {candidate}")

        self.model_info_path = candidate
        if not self.model_directory:
            self.model_directory = os.path.dirname(candidate)

    def load(
        self, extractor: Optional[Callable[[dict], Dict[int, str]]] = None
    ) -> None:
        """Loads class mapping and computes water-related class indices.

        Args:
            extractor (Optional[Callable[[dict], Dict[int, str]]]): Optional function
                that extracts class mappings from model card data.
        """
        self.class_mapping = self._load_class_mapping(
            extractor=extractor,
            saved_model_info=None,
        )
        self.water_class_indices = self._resolve_water_classes(None)

    def load_from_model_info_file(
        self,
        model_info_path: Optional[str] = None,
    ) -> None:
        """Load model metadata from a ``model_info.json`` file.

        Args:
            model_info_path (Optional[str]): Full path to ``model_info.json``.

        Raises:
            FileNotFoundError: If ``model_info.json`` cannot be found.
            ValueError: If file content is not a valid JSON object.
        """
        resolved_path: Optional[str] = None
        if model_info_path and model_info_path.strip():
            resolved_path = os.path.abspath(model_info_path)
        elif self.model_info_path and self.model_info_path.strip():
            resolved_path = os.path.abspath(self.model_info_path)

        if not resolved_path or not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"model_info.json file not found: {resolved_path or '<unspecified>'}"
            )

        try:
            with open(resolved_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            raise ValueError(
                f"Failed to read model_info.json: {resolved_path}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"model_info.json must contain a JSON object: {resolved_path}"
            )

        self.model_info_path = resolved_path
        self.model_directory = os.path.dirname(resolved_path)
        # Explicit file loads should replace any previously loaded/default mapping.
        self.class_mapping = {}
        self.water_class_indices = []
        self.class_mapping = self._load_class_mapping(saved_model_info=payload)
        self.water_class_indices = self._resolve_water_classes(payload)

        input_directory = payload.get("input_directory")
        if isinstance(input_directory, str) and input_directory.strip():
            self.input_directory = input_directory

    def _resolve_directory(self) -> Optional[str]:
        """Resolves the model directory from inputs.

        Returns:
            Optional[str]: Existing model directory path, or None when no model source
            is provided.

        Raises:
            FileNotFoundError: If a provided or resolved model directory does not exist.
        """
        if self.model_directory:
            path = self.model_directory
        elif self.model_name:
            path = os.path.join(common.get_downloaded_models_dir(), self.model_name)
        else:
            return None

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        return path

    def _load_class_mapping(
        self,
        extractor: Optional[Callable[[dict], Dict[int, str]]] = None,
        saved_model_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """Loads class mapping from memory, model card, or defaults.

        Args:
            extractor (Optional[Callable[[dict], Dict[int, str]]]): Optional parser
                for model card JSON content.

        Example class mapping:
            {
                0: "water",
                1: "whitewater",
                2: "sediment",
                3: "other"
            }

        Returns:
            Dict[int, str]: Mapping of class index to class label.
        """
        if self.class_mapping:
            return self.class_mapping

        if isinstance(saved_model_info, dict):
            class_mapping = saved_model_info.get("class_mapping")
            if isinstance(class_mapping, dict):
                resolved_mapping: Dict[int, str] = {}
                for key, value in class_mapping.items():
                    try:
                        resolved_mapping[int(key)] = str(value)
                    except (TypeError, ValueError):
                        continue
                if resolved_mapping:
                    return resolved_mapping

        if not self.model_directory:
            return DEFAULT_CLASS_MAPPING.copy()

        try:
            # Find the model card JSON file to read the class mapping.
            path = file_utilities.find_file_by_regex(
                self.model_directory, r".*modelcard\.json$"
            )
            model_card = file_utilities.read_json_file(path, raise_error=True)
            # load the class mapping using the provided extractor function if available
            if extractor:
                return extractor(model_card)
            # load the class mapping from the model card using the default extractor
            return self._default_extractor(model_card)
        except Exception:
            return DEFAULT_CLASS_MAPPING.copy()

    def _resolve_water_classes(
        self,
        saved_model_info: Optional[Dict[str, Any]],
    ) -> List[int]:
        """Resolve water-class indices from saved metadata or class labels.

        Args:
            saved_model_info (Optional[Dict[str, Any]]): Parsed model_info payload.

        Returns:
            List[int]: Water class indices.
        """
        if isinstance(saved_model_info, dict):
            saved_indices = saved_model_info.get("water_class_indices")
            if isinstance(saved_indices, list):
                resolved_indices: List[int] = []
                for value in saved_indices:
                    try:
                        resolved_indices.append(int(value))
                    except (TypeError, ValueError):
                        continue
                if resolved_indices:
                    return sorted(set(resolved_indices))

            saved_names = saved_model_info.get("water_classes")
            if isinstance(saved_names, list):
                normalized_names = {str(name).strip().lower() for name in saved_names}
                resolved = [
                    class_index
                    for class_index, class_name in self.class_mapping.items()
                    if str(class_name).strip().lower() in normalized_names
                ]
                if resolved:
                    return sorted(set(resolved))

        return self._find_water_classes()

    def _find_water_classes(
        self, water_names: List[str] = ["water", "whitewater"]
    ) -> List[int]:
        """Finds class indices whose labels represent water.

        Args:
            water_names (List[str]): Labels treated as water classes.

        Returns:
            List[int]: Indices matching any label in ``water_names``.
        """
        return [i for i, name in self.class_mapping.items() if name in water_names]

    @staticmethod
    def _default_extractor(data: dict) -> Dict[int, str]:
        """Extracts class mapping from standard model card sections.

        Example:
        {
        "DATASET": {
            "CLASSES": {
                "0": "water",
                "1": "whitewater",
                "2": "sediment",
                "3": "other"
            }

        }
        }
        Args:
            data (dict): Parsed model card JSON.

        Returns:
            Dict[int, str]: Extracted mapping or the default class mapping when no
            supported section is found.
        """
        for section in ("DATASET", "DATASET1"):
            if section in data and "CLASSES" in data[section]:
                return {int(k): v for k, v in data[section]["CLASSES"].items()}
        if "CLASSES" in data:
            return {int(k): v for k, v in data["CLASSES"].items()}
        return DEFAULT_CLASS_MAPPING.copy()
