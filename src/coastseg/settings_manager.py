"""
Settings Management System for CoastSeg

This module provides a robust, extensible settings management system using
dataclasses for type safety, validation, and easy serialization.
"""

import logging
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints
import json
from dateutil import parser as date_parser

from . import __version__

logger = logging.getLogger(__name__)


class SettingsError(Exception):
    """Raised when there's an error with settings validation or processing."""
    pass


@dataclass
class DownloadSettings:
    """Settings for downloading satellite imagery."""
    landsat_collection: str = "C02"
    dates: List[str] = field(default_factory=lambda: ["2017-12-01", "2018-01-01"])
    months_list: List[int] = field(default_factory=lambda: list(range(1, 13)))
    sat_list: List[str] = field(default_factory=lambda: ["L8"])
    download_cloud_thresh: float = 0.8
    min_roi_coverage: float = 0.5
    
    def __post_init__(self):
        """Validate settings after initialization."""
        self._validate_dates()
        self._validate_satellites()
        self._validate_thresholds()
    
    def _validate_dates(self):
        """Validate and normalize date format."""
        if len(self.dates) != 2:
            raise SettingsError("dates must contain exactly 2 elements [start_date, end_date]")
        
        normalized_dates = []
        for date_str in self.dates:
            try:
                if isinstance(date_str, datetime):
                    normalized_dates.append(date_str.strftime("%Y-%m-%d"))
                else:
                    # Parse flexible date formats
                    parsed_date = date_parser.parse(date_str)
                    normalized_dates.append(parsed_date.strftime("%Y-%m-%d"))
            except (ValueError, TypeError) as e:
                raise SettingsError(f"Invalid date format: {date_str}") from e
        
        self.dates = normalized_dates
    
    def _validate_satellites(self):
        """Validate satellite list."""
        valid_satellites = {"L5", "L7", "L8", "L9", "S2", "PS"}
        invalid_sats = set(self.sat_list) - valid_satellites
        if invalid_sats:
            raise SettingsError(f"Invalid satellites: {invalid_sats}")
    
    def _validate_thresholds(self):
        """Validate threshold values."""
        if not 0 <= self.download_cloud_thresh <= 1:
            raise SettingsError("download_cloud_thresh must be between 0 and 1")
        if not 0 <= self.min_roi_coverage <= 1:
            raise SettingsError("min_roi_coverage must be between 0 and 1")


@dataclass
class ProcessingSettings:
    """Settings for image processing and shoreline extraction."""
    cloud_thresh: float = 0.8
    percent_no_data: float = 0.8
    dist_clouds: int = 300
    output_epsg: int = 4326
    check_detection: bool = False
    adjust_detection: bool = False
    save_figure: bool = True
    min_beach_area: int = 4500
    min_length_sl: int = 100
    cloud_mask_issue: bool = False
    sand_color: str = "default"
    pan_off: str = "False"
    apply_cloud_mask: bool = True
    image_size_filter: bool = True
    
    def __post_init__(self):
        """Validate processing settings."""
        self._validate_thresholds()
        self._validate_epsg()
    
    def _validate_thresholds(self):
        """Validate threshold values."""
        thresholds = {
            'cloud_thresh': self.cloud_thresh,
            'percent_no_data': self.percent_no_data
        }
        for name, value in thresholds.items():
            if not 0 <= value <= 1:
                raise SettingsError(f"{name} must be between 0 and 1")
    
    def _validate_epsg(self):
        """Validate EPSG code."""
        if self.output_epsg <= 0:
            raise SettingsError("output_epsg must be a positive integer")


@dataclass
class TransectSettings:
    """Settings for transect analysis."""
    max_dist_ref: int = 25
    along_dist: int = 25
    min_points: int = 3
    max_std: int = 15
    max_range: int = 30
    min_chainage: int = -100
    multiple_inter: str = "auto"
    prc_multiple: float = 0.1
    drop_intersection_pts: bool = False
    
    def __post_init__(self):
        """Validate transect settings."""
        if self.multiple_inter not in ["auto", "nan", "max"]:
            raise SettingsError("multiple_inter must be one of: 'auto', 'nan', 'max'")
        if not 0 <= self.prc_multiple <= 1:
            raise SettingsError("prc_multiple must be between 0 and 1")


@dataclass
class CoastSegSettings:
    """Master settings container for CoastSeg application."""
    download: DownloadSettings = field(default_factory=DownloadSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    transect: TransectSettings = field(default_factory=TransectSettings)
    coastseg_version: str = field(default_factory=lambda: __version__)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoastSegSettings':
        """Create settings from dictionary with flexible key handling."""
        # Separate settings by category
        download_keys = {f.name for f in fields(DownloadSettings)}
        processing_keys = {f.name for f in fields(ProcessingSettings)}
        transect_keys = {f.name for f in fields(TransectSettings)}
        
        download_data = {k: v for k, v in data.items() if k in download_keys}
        processing_data = {k: v for k, v in data.items() if k in processing_keys}
        transect_data = {k: v for k, v in data.items() if k in transect_keys}
        
        # Handle coastseg_version separately
        version = data.get('coastseg_version', __version__)
        
        return cls(
            download=DownloadSettings(**download_data),
            processing=ProcessingSettings(**processing_data),
            transect=TransectSettings(**transect_data),
            coastseg_version=version
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to flat dictionary for backward compatibility."""
        result = {}
        result.update(asdict(self.download))
        result.update(asdict(self.processing))
        result.update(asdict(self.transect))
        result['coastseg_version'] = self.coastseg_version
        return result
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update settings from dictionary while preserving validation."""
        new_settings = self.from_dict({**self.to_dict(), **data})
        self.download = new_settings.download
        self.processing = new_settings.processing
        self.transect = new_settings.transect
        self.coastseg_version = new_settings.coastseg_version
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save settings to JSON file."""
        filepath = Path(filepath)
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            logger.info(f"Settings saved to {filepath}")
        except IOError as e:
            raise SettingsError(f"Failed to save settings to {filepath}") from e
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'CoastSegSettings':
        """Load settings from JSON file."""
        filepath = Path(filepath)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Settings loaded from {filepath}")
            return cls.from_dict(data)
        except IOError as e:
            raise SettingsError(f"Failed to load settings from {filepath}") from e
        except json.JSONDecodeError as e:
            raise SettingsError(f"Invalid JSON in settings file {filepath}") from e
    
    def validate_for_download(self) -> None:
        """Validate that all required settings for download are present and valid."""
        required_fields = ['dates', 'sat_list', 'landsat_collection']
        missing = []
        
        for field_name in required_fields:
            if not hasattr(self.download, field_name):
                missing.append(field_name)
            elif getattr(self.download, field_name) in [None, [], ""]:
                missing.append(field_name)
        
        if missing:
            raise SettingsError(f"Missing required download settings: {missing}")
    
    def get_roi_specific_settings(self, roi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ROI-specific settings by merging global settings with ROI data."""
        base_settings = self.to_dict()
        
        # Add ROI-specific fields
        roi_settings = {
            **base_settings,
            'sitename': roi_data.get('id', 'unknown'),
            'polygon': roi_data.get('geometry', {}),
            'filepath_data': roi_data.get('filepath_data', ''),
            'roi_id': roi_data.get('id', ''),
        }
        
        return roi_settings


class SettingsManager:
    """
    Centralized settings manager for CoastSeg application.
    
    Provides a clean interface for settings management with validation,
    serialization, and backward compatibility.
    """
    
    def __init__(self, settings: Optional[CoastSegSettings] = None):
        """Initialize settings manager."""
        self._settings = settings or CoastSegSettings()
        self._observers = []  # For future observer pattern implementation
    
    @property
    def settings(self) -> CoastSegSettings:
        """Get current settings."""
        return self._settings
    
    def update_settings(self, **kwargs) -> None:
        """Update settings with validation."""
        try:
            self._settings.update_from_dict(kwargs)
            logger.info(f"Settings updated: {list(kwargs.keys())}")
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            raise SettingsError(f"Settings update failed: {e}") from e
    
    def get_flat_dict(self) -> Dict[str, Any]:
        """Get settings as flat dictionary for backward compatibility."""
        return self._settings.to_dict()
    
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """Load settings from file."""
        self._settings = CoastSegSettings.load_from_file(filepath)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save settings to file."""
        self._settings.save_to_file(filepath)
    
    def create_roi_settings(self, roi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create settings for a specific ROI."""
        return self._settings.get_roi_specific_settings(roi_data)
    
    def __str__(self) -> str:
        """String representation of settings."""
        return f"SettingsManager(version={self._settings.coastseg_version})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"SettingsManager(\n"
                f"  download={self._settings.download},\n"
                f"  processing={self._settings.processing},\n"
                f"  transect={self._settings.transect}\n"
                f")")
