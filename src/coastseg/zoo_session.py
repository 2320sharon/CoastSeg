import os
import json
from pathlib import Path
from typing import Optional, Dict
import geopandas as gpd


class ZooSession:
    """
    Manages the session context for Zoo model processing including paths, config, and basic validation.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.session_path = Path(config.get("model_session_path", "")).resolve()
        self.sitename = config.get("sitename", "")
        self.sample_dir = Path(config.get("sample_direc", "")).resolve() if config.get("sample_direc") else None

    def validate(self) -> None:
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session path does not exist: {self.session_path}")
        if not self.sitename:
            raise ValueError("Missing 'sitename' in session configuration.")

    @property
    def rgb_dir(self) -> Path:
        return self.session_path / "jpg_files" / "preprocessed" / "RGB"

    @property
    def nir_dir(self) -> Path:
        return self.session_path / "jpg_files" / "preprocessed" / "NIR"

    @property
    def swir_dir(self) -> Path:
        return self.session_path / "jpg_files" / "preprocessed" / "SWIR"

    @property
    def metadata_file(self) -> Path:
        return self.session_path / f"{self.sitename}_metadata.json"

    @property
    def transects_geojson(self) -> Optional[gpd.GeoDataFrame]:
        transect_path = self.session_path / "transects" / "transects.geojson"
        return gpd.read_file(transect_path) if transect_path.exists() else None

    def load_metadata(self) -> Optional[Dict]:
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return None

    def to_dict(self) -> Dict:
        return {
            "session_path": str(self.session_path),
            "sitename": self.sitename,
            "sample_dir": str(self.sample_dir) if self.sample_dir else None,
            "rgb_dir": str(self.rgb_dir),
            "metadata_file": str(self.metadata_file),
        }
