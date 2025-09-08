"""
Minimal Factory Pattern for CoastSeg Features

This module provides a lightweight, type-safe factory for creating coastal feature objects
without dependencies on heavyweight classes like CoastSeg_Map. The factory uses enums for
type safety and delegates creation logic to the feature classes themselves via @classmethods.

REFACTORING REASONING:
=====================

Previous Problems:
1. Factory.make_feature() required a CoastSeg_Map instance as a parameter
2. This created tight coupling between the factory and the massive CoastSeg_Map class
3. Factory stored references to heavyweight objects, causing memory overhead
4. String-based feature names were error-prone and not type-safe
5. Complex creation logic was mixed into the factory instead of being encapsulated in feature classes

New Approach Benefits:
1. NO DEPENDENCY on CoastSeg_Map - eliminates tight coupling and reduces memory usage
2. TYPE-SAFE enums instead of error-prone string keys - better IDE support and fewer bugs  
3. MINIMAL memory footprint - just mapping functions, no stored state or heavy objects
4. FEATURE CLASSES handle their own creation logic via @classmethods - better separation of concerns
5. CLEANER API - simple functions instead of heavyweight classes
6. BACKWARD COMPATIBILITY maintained via create_feature_from_string() function

Key improvements over the previous approach:
- No dependency on CoastSeg_Map instances - eliminates tight coupling
- Type-safe enum-based feature types instead of error-prone strings
- Minimal memory footprint - just a mapping function, no stored state
- Feature classes handle their own creation logic via @classmethods
- Cleaner separation of concerns
"""

import logging
from typing import Union, Optional, Type
from enum import Enum
from geopandas import GeoDataFrame
from math import sqrt

# Import feature classes - these now handle their own creation logic
from coastseg.shoreline_extraction_area import Shoreline_Extraction_Area
from coastseg.bbox import Bounding_Box
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI

logger = logging.getLogger(__name__)

__all__ = ["FeatureType", "create_feature"]


class FeatureType(Enum):
    """
    Type-safe enumeration of supported coastal feature types.
    
    Using enums instead of strings prevents typos and provides better IDE support
    with autocomplete and type checking.
    """
    SHORELINE = "shoreline"
    TRANSECTS = "transects"
    BBOX = "bbox"
    ROIS = "rois"
    SHORELINE_EXTRACTION_AREA = "shoreline_extraction_area"


# Minimal mapping from enum types to feature classes
# This is much cleaner than a dictionary of strings and eliminates the need
# for a heavyweight Factory class that stores state
_FEATURE_CLASS_MAP = {
    FeatureType.SHORELINE: Shoreline,
    FeatureType.TRANSECTS: Transects,
    FeatureType.BBOX: Bounding_Box,
    FeatureType.ROIS: ROI,
    FeatureType.SHORELINE_EXTRACTION_AREA: Shoreline_Extraction_Area,
}


def create_feature(
    feature_type: FeatureType,
    gdf: Optional[GeoDataFrame] = None,
    **kwargs
) -> Union[Shoreline, Transects, Bounding_Box, ROI, Shoreline_Extraction_Area]:
    """
    Create a coastal feature object using a minimal factory pattern.
    
    This function replaces the heavyweight Factory class with a simple function
    that delegates to feature class constructors or @classmethods.
    
    Args:
        feature_type: Type-safe enum specifying which feature to create
        gdf: Optional GeoDataFrame to initialize the feature with
        **kwargs: Additional arguments passed to the feature constructor
            bbox: Optional bounding box GeoDataFrame to load the feature within

    Returns:
        Instance of the requested feature type, or None if gdf is empty
        
    Raises:
        ValueError: If feature_type is not supported
        ValueError: If the provided GeoDataFrame is empty

    Example:
        # Create a shoreline from a GeoDataFrame
        shoreline = create_feature(FeatureType.SHORELINE, gdf=my_gdf)
        
        # Create transects from a bounding box (uses @classmethod)
        transects = create_feature(FeatureType.TRANSECTS, bbox=bbox_gdf)
    """
    if feature_type not in _FEATURE_CLASS_MAP:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    feature_class = _FEATURE_CLASS_MAP[feature_type]
    
    # Handle empty GeoDataFrame case
    if gdf is not None and gdf.empty:
        logger.warning(f"Empty GeoDataFrame provided for {feature_type.value}")
        raise ValueError(f"Provided GeoDataFrame for {feature_type.value} is empty")
    logger.info(f"Creating {feature_type.value} feature")
    #@todo just call the class constructor no fancy business.
    #@todo the factory should not be handling any gdf not found right????

    # all of these are correct when only a gdf is passed
    # For simple cases where we have a GeoDataFrame, use the constructor directly
    if gdf is not None:
        if feature_type == FeatureType.SHORELINE:
            return feature_class(shoreline=gdf, **kwargs)
        elif feature_type == FeatureType.TRANSECTS:
            return feature_class(transects=gdf, **kwargs) # checked this is correct - kwargs
        elif feature_type == FeatureType.BBOX:
            return feature_class(gdf, **kwargs)
        elif feature_type == FeatureType.ROIS:
            return feature_class(rois_gdf=gdf, **kwargs)
        elif feature_type == FeatureType.SHORELINE_EXTRACTION_AREA:
            return feature_class(gdf, **kwargs)
    
    # For complex cases (like creating from bounding box), use @classmethods
    # These are now available on the feature classes
    else:
        if feature_type == FeatureType.SHORELINE and 'bbox' in kwargs:
            return feature_class.from_bbox(kwargs['bbox'], **{k: v for k, v in kwargs.items() if k != 'bbox'})
        elif feature_type == FeatureType.TRANSECTS and 'bbox' in kwargs:
            return feature_class.from_bbox(kwargs['bbox'], **{k: v for k, v in kwargs.items() if k != 'bbox'})
        elif feature_type == FeatureType.BBOX and 'geojson_dict' in kwargs:
            return feature_class.from_geojson_dict(kwargs['geojson_dict'], **{k: v for k, v in kwargs.items() if k != 'geojson_dict'})
        elif feature_type == FeatureType.ROIS:

            lg_area = kwargs.get("lg_area")
            sm_area = kwargs.get("sm_area")
            units = kwargs.get("units")
            if lg_area is None or sm_area is None or units is None:
                raise Exception("Must provide ROI area and units")

            # if units is kilometers convert to meters
            if units == "km²":
                sm_area = sm_area * (10**6)
                lg_area = lg_area * (10**6)

            # get length of ROI square by taking square root of area
            small_len = sqrt(sm_area)
            large_len = sqrt(lg_area)
            return feature_class(bbox=kwargs['bbox'], shoreline=kwargs['shoreline'], small_len=small_len, large_len=large_len)
        else:
            # Fallback to constructor with kwargs
            return feature_class(**kwargs)


# Convenience functions for backward compatibility and simpler usage
def create_shoreline(gdf: Optional[GeoDataFrame] = None, **kwargs) -> Optional[Shoreline]:
    """Create a Shoreline feature with optional GeoDataFrame."""
    result = create_feature(FeatureType.SHORELINE, gdf=gdf, **kwargs)
    return result if isinstance(result, Shoreline) else None

#@todo unused remove later
def create_transects(gdf: Optional[GeoDataFrame] = None, **kwargs) -> Optional[Transects]:
    """Create a Transects feature with optional GeoDataFrame."""
    result = create_feature(FeatureType.TRANSECTS, gdf=gdf, **kwargs)
    return result if isinstance(result, Transects) else None

#@todo unused remove later
def create_bbox(gdf: Optional[GeoDataFrame] = None, **kwargs) -> Optional[Bounding_Box]:
    """Create a Bounding_Box feature with optional GeoDataFrame."""
    result = create_feature(FeatureType.BBOX, gdf=gdf, **kwargs)
    return result if isinstance(result, Bounding_Box) else None

#@todo unused remove later
def create_rois(gdf: Optional[GeoDataFrame] = None, **kwargs) -> Optional[ROI]:
    """Create an ROI feature with optional GeoDataFrame."""
    result = create_feature(FeatureType.ROIS, gdf=gdf, **kwargs)
    return result if isinstance(result, ROI) else None

#@todo unused remove later
def create_shoreline_extraction_area(gdf: Optional[GeoDataFrame] = None, **kwargs) -> Optional[Shoreline_Extraction_Area]:
    """Create a Shoreline_Extraction_Area feature with optional GeoDataFrame."""
    result = create_feature(FeatureType.SHORELINE_EXTRACTION_AREA, gdf=gdf, **kwargs)
    return result if isinstance(result, Shoreline_Extraction_Area) else None


# Legacy compatibility - maintain the string-based interface for existing code
# but map to the new enum-based system internally
def create_feature_from_string(
    feature_name: str,
    gdf: Optional[GeoDataFrame] = None,
    **kwargs
) -> Union[Shoreline, Transects, Bounding_Box, ROI, Shoreline_Extraction_Area]:
    """
    Legacy compatibility function for string-based feature creation.
    
    This maintains backward compatibility while internally using the new
    type-safe enum system.
    """
    # Map common string names to enums
    string_to_enum = {
        "shoreline": FeatureType.SHORELINE,
        "shorelines": FeatureType.SHORELINE,
        "reference_shoreline": FeatureType.SHORELINE,
        "reference shorelines": FeatureType.SHORELINE,
        "reference shoreline": FeatureType.SHORELINE,
        "reference_shorelines": FeatureType.SHORELINE,
        "transects": FeatureType.TRANSECTS,
        "transect": FeatureType.TRANSECTS,
        "bbox": FeatureType.BBOX,
        "rois": FeatureType.ROIS,
        "roi": FeatureType.ROIS,
        "shoreline_extraction_area": FeatureType.SHORELINE_EXTRACTION_AREA,
        "shoreline extraction area": FeatureType.SHORELINE_EXTRACTION_AREA,
    }
    
    feature_type = string_to_enum.get(feature_name.lower())
    if feature_type is None:
        raise ValueError(f"Unknown feature name: {feature_name}")
    
    return create_feature(feature_type, gdf=gdf, **kwargs)
