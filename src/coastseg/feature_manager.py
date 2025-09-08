from typing import Optional, Any, Callable

import geopandas as gpd
from geopandas import GeoDataFrame

from build.lib.coastseg import exception_handler
from coastseg.factory import FeatureType, create_feature_from_string,create_feature
from coastseg.transects import Transects
from coastseg.exceptions import Object_Not_Found

def merge_rectangles(gdf: GeoDataFrame) -> GeoDataFrame:
    """
    Merges all rectangles in a GeoDataFrame into a single shape.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the rectangles.

    Returns:
        GeoDataFrame: A new GeoDataFrame containing a single shape that is the union of all rectangles in gdf.
                          The new GeoDataFrame has the same columns as the original.
    """
    # Ensure that the GeoDataFrame contains Polygons
    if not all(gdf.geometry.geom_type == "Polygon"):
        raise ValueError("All shapes in the GeoDataFrame must be Polygons.")

    # Merge all shapes into one
    merged_shape = gdf.union_all()

    # Create a new GeoDataFrame with the merged shape and the same columns as the original
    merged_gdf = GeoDataFrame([gdf.iloc[0]], geometry=[merged_shape], crs=gdf.crs)

    return merged_gdf

def _is_nonempty_gdf(gdf: Optional[gpd.GeoDataFrame]) -> bool:
    return gdf is not None and not gdf.empty

def _validate_feature_exists(feature: Optional[gpd.GeoDataFrame], feature_name: str, exception_handler) -> None:
    # Call your exception checks only for the bbox path
    exception_handler.check_if_None(feature, feature_name)
    exception_handler.check_if_gdf_empty(feature, feature_name)

class FeatureManager:
    def __init__(self):
        self.features = {}  # {FeatureType: feature instance}
        self.gdfs = {}      # {'rois': gdf, 'bbox': gdf, ...}
        self.listeners = [] # Add listeners to handle features being unloaded and loaded

    def add_listener(self, callback):
        """Function to run when any event occurs."""
        self.listeners.append(callback)

    def notify(self, event, feature_type):
        """Notify all listeners of an event and the feature type that emitted it"""
        for callback in self.listeners:
            callback(event, feature_type)

    def register_gdf(self, name, gdf):
        """Register a geodataframe (e.g., 'rois', 'bbox') for filtering."""
        self.gdfs[name] = gdf

    def is_loaded(self, feature_type: FeatureType) -> bool:
        """Check if a feature of the given type is loaded."""
        return feature_type in self.features

    def get_feature(self, feature_type: FeatureType) -> Optional[Any]:
        """Retrieve a loaded feature by type."""
        return self.features.get(feature_type)

    def unload_feature(self, feature_type: FeatureType) -> None:
        """Remove a feature from the manager."""
        self.features.pop(feature_type, None)
        self.notify('removed', feature_type)

    def get_gdf(self, name):
        """Retrieve a registered GeoDataFrame (gdf) by name."""
        return self.gdfs.get(name)

    def send_notification_for_feature(self, feature_type: FeatureType, **kwargs) -> Optional[Any]:
        if feature_type == FeatureType.SHORELINE:
            # unsure if we need notify that the old shoreline needs to be removed?
            self.notify('removed', FeatureType.SHORELINE) #??? @todo do I need this?
        elif feature_type == FeatureType.BBOX:
            #make sure to notify listeners because bbox needs to be removed from the map after new one is loaded
            self.notify('removed', FeatureType.BBOX) # coastsegmap.remove_bbox(), coastsegmap.draw_control.clear(), coastsegmap.bbox = bbox 
            # this should be created as normal from the gdf
        elif feature_type == FeatureType.ROIS:
            pass
        elif feature_type == FeatureType.SHORELINE_EXTRACTION_AREA:
            self.notify('removed', FeatureType.SHORELINE_EXTRACTION_AREA) # coastsegmap.remove_shoreline_extraction_area(), coastsegmap.draw_control.clear(), coastsegmap.shoreline_extraction_area = shoreline_extraction_area
        elif feature_type == FeatureType.TRANSECTS:
            self.notify('removed', FeatureType.TRANSECTS) #???@todo do I need this?
        else:
            raise ValueError(f"No loading strategy for feature type: {feature_type}")

    def load_feature(
        self,
        feature_type: FeatureType,
        *,
        gdf: Optional[gpd.GeoDataFrame] = None,
        file: str = "",
        strategy: Optional[Callable[['FeatureManager'], gpd.GeoDataFrame]] = None,
        **kwargs
    ) -> Any:
        """
        Load a feature using the provided strategy or default logic.
        Priority:
            1. User-provided GeoDataFrame
            2. User-provided file
            3. Custom strategy (if provided)
            4. Default strategies (ROIs > BBox > all)
        """
        # @todo convert all these to create_feature
        # 1. User-provided GeoDataFrame
        if gdf is not None:
            feature = create_feature_from_string(feature_type.value, gdf=gdf, **kwargs)
        # 2. User-provided file
        elif file:
            gdf_from_file = gpd.read_file(file)
            feature = create_feature_from_string(feature_type.value, gdf=gdf_from_file, **kwargs)
        # 3.Custom strategies for specific features
        else:
            if feature_type == FeatureType.SHORELINE:
                feature = self.shoreline_creation_strategy(**kwargs)
            elif feature_type == FeatureType.ROIS:
                pass
            elif feature_type == FeatureType.TRANSECTS:
                feature = self.transect_creation_strategy(**kwargs)
            else:
                raise ValueError(f"The provided file or geodataframe was empty or not available for the feature {feature_type}")

        self.send_notification_for_feature(feature_type)
        #@todo convert all these to create_feature
        # # 1. User-provided GeoDataFrame
        # if gdf is not None:
        #     feature = create_feature_from_string(feature_type.value, gdf=gdf, **kwargs)
        # # 2. User-provided file
        # elif file:
        #     gdf_from_file = gpd.read_file(file)
        #     feature = create_feature_from_string(feature_type.value, gdf=gdf_from_file, **kwargs)
        # # 3. Custom strategy
        # elif strategy is not None:
        #     gdf_strategy = strategy(self)
        #     feature = create_feature_from_string(feature_type.value, gdf=gdf_strategy, **kwargs)
        # # 4. Default strategies for transects
        # elif feature_type == FeatureType.TRANSECTS:
        #     feature = self.transect_creation_strategy(**kwargs)
        # else:
        #     raise ValueError(f"No loading strategy for feature type: {feature_type}")

        self.features[feature_type] = feature
        self.notify('loaded', feature_type) # notify listeners of the loaded feature
        return feature
    
    def ROI_creation_strategy(self, **kwargs):
        """
        Create 'ROIs' feature based on available inputs.

        Priority:
        1) If ROIs exist (non-empty), merge them and make transects from that.
        2) Else, require a valid bbox; intersect all_transects with bbox.

        Raises:
        Whatever your exception_handler raises when bbox is missing/empty.
        """
        shorelines = self.get_gdf('shorelines')
        bbox = self.get_gdf('bbox')
        # 1. Require a valid bbox to load ROIs within
        _validate_feature_exists(bbox, "bounding box", exception_handler)
        # 2. If a shoreline does not exist load one within the bbox
        if not _is_nonempty_gdf(shorelines):
            try:
                shorelines = create_feature(FeatureType.SHORELINE,bbox=bbox, **kwargs)
            except Exception as e:
                raise Object_Not_Found("shoreline", "Cannot create an ROI without a shoreline. No shorelines were available in this region. Please upload a shoreline from a file")
            # if shorelines could not be created
            # notify that shorelines were created and should be loaded on the map
            self.notify('loaded', FeatureType.SHORELINE) # @todo shouldn't we pass the feature so we can load it onto the map?

        transects = create_feature(FeatureType.TRANSECTS,bbox=bbox, **kwargs)

        exception_handler.check_if_default_feature_available(transects.gdf, "transects")
        return transects


    def transect_creation_strategy(self, **kwargs):
        """
        Create 'transects' feature based on available inputs.

        Priority:
        1) If ROIs exist (non-empty), merge them and make transects from that.
        2) Else, require a valid bbox; intersect all_transects with bbox.

        Raises:
        Whatever your exception_handler raises when bbox is missing/empty.
        """
        rois = self.get_gdf('rois')
        bbox = self.get_gdf('bbox')

        # 1. Prefer to load transects within ROIs if available
        if _is_nonempty_gdf(rois):
            # merge ROI geometeries together and use that as the bbox to load the transects within
            merged_rois = merge_rectangles(rois)
            transects = create_feature(FeatureType.TRANSECTS,bbox=merged_rois, **kwargs)
        else: # 2. Otherwise, require a valid bbox to load transects within
            # Handle cases where both the ROIs and the bbox are not loaded on the map
            _validate_feature_exists(bbox, "bounding box", exception_handler)
            transects = create_feature(FeatureType.TRANSECTS,bbox=bbox, **kwargs)

        exception_handler.check_if_default_feature_available(transects.gdf, "transects")
        return transects

    def shoreline_creation_strategy(self, **kwargs):
        """
        Create 'shoreline' feature based on available inputs.
        
        Priority:
        1) If ROIs exist (non-empty), merge to geometries into a single geometry and create shorelines within the merged ROIs.
        2) Else, require a valid bbox; create all the shorelines within the bbox

        Raises:
        Whatever your exception_handler raises when bbox is missing/empty.
        """

        rois = self.get_gdf('rois')
        bbox = self.get_gdf('bbox')

        # 1. Prefer to load shorelines within ROIs if available
        if _is_nonempty_gdf(rois):
            # merge ROI geometeries together and use that as the bbox to load the shorelines within
            merged_rois = merge_rectangles(rois)
            shorelines = create_feature(FeatureType.SHORELINE,bbox=merged_rois, **kwargs)
        else: # 2. Otherwise, require a valid bbox to load shorelines within
            # Handle cases where both the ROIs and the bbox are not loaded on the map
            _validate_feature_exists(bbox, "bounding box", exception_handler)
            shorelines = create_feature(FeatureType.SHORELINE,bbox=bbox, **kwargs)

        exception_handler.check_if_default_feature_available(shorelines.gdf, "shorelines")
        return shorelines
    

#@todo remove later
# # --- Example usage for transects ---

# if __name__ == "__main__":
#     manager = FeatureManager()

#     # Register spatial context (ROIs and BBox)
#     rois_gdf = gpd.read_file("rois.geojson")
#     bbox_gdf = gpd.read_file("bbox.geojson")
#     manager.register_gdf('rois', rois_gdf)
#     manager.register_gdf('bbox', bbox_gdf)

#     # 1. Load transects from a user-provided file
#     transects_from_file = manager.load_feature(
#         FeatureType.TRANSECTS,
#         file="user_transects.geojson"
#     )

#     # 2. Load transects from a user-provided GeoDataFrame
#     user_gdf = gpd.read_file("custom_transects.geojson")
#     transects_from_gdf = manager.load_feature(
#         FeatureType.TRANSECTS,
#         gdf=user_gdf
#     )

#     # 3. Load transects within ROIs (default strategy)
#     transects_in_rois = manager.load_feature(
#         FeatureType.TRANSECTS
#     )

#     # 4. Check what's loaded
#     print("Loaded features:", list(manager.features.keys()))
#     print("Transects loaded:", manager.is_loaded(FeatureType.TRANSECTS))
#     print("Transects object:", manager.get_feature(FeatureType.TRANSECTS))