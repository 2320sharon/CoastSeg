from abc import ABC
from ipyleaflet import GeoJSON
import geopandas as gpd
import json

class Feature(ABC):

    def style_layer(self, data: dict, layer_name: str,style:dict={},hover_style:dict={}) -> GeoJSON:
        """
        Styles and returns a GeoJSON layer for visualization on a map.
        Default style is a gray solid line and red on hover.
        
        Parameters
        ----------
        data : dict or geopandas.GeoDataFrame
            The geographic data to be styled and displayed. Can be a GeoJSON dictionary or a GeoDataFrame.
        layer_name : str
            The name to assign to the layer on the map.
        style : dict, optional
            A dictionary specifying the style properties for the layer (e.g., color, fill color, opacity, weight).
            If not provided, a default style is used.
        hover_style : dict, optional
            A dictionary specifying the style properties to apply when hovering over the layer.
            If not provided, no hover style is applied.
        Returns
        -------
        GeoJSON
            A styled GeoJSON layer ready to be added to a map.
        Raises
        ------
        AssertionError
            If the input data is empty and cannot be drawn onto the map.
        """
        if isinstance(data, dict):
            geojson = data
        elif isinstance(data,gpd.GeoDataFrame):
            geojson  = json.loads(data.to_json())
 
        assert (
            geojson != {}
        ), f"ERROR.\n Empty {layer_name} geojson cannot be drawn onto map"
        if not hover_style:
            hover_style={}
    
        # Set a default style if none is provided
        if not style:
            style={
                "color": "#555555",
                "fill_color": "#555555",
                "fillOpacity": 0.1,
                "weight": 1,
            }
        
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style=style,
            hover_style=hover_style,
        )