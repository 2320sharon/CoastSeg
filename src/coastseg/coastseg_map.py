import os
import json
import logging
import copy
from typing import Union

from coastseg.bbox import Bounding_Box
from coastseg import common
from coastseg import factory
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg import exceptions
from coastseg import extracted_shoreline
from coastseg import exception_handler

from coastsat import (
    SDS_download,
    SDS_transects,
    SDS_preprocess,
)
import geopandas as gpd
from ipyleaflet import DrawControl, LayersControl, WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML, Accordion
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class CoastSeg_Map:
    def __init__(self, settings: dict = None):
        self.factory = factory.Factory()
        # settings:  used to select data to download and preprocess settings
        self.settings = {
            # general parameters:
            "cloud_thresh": 0.5,  # threshold on maximum cloud cover
            "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
            "output_epsg": 4326,  # epsg code of spatial reference system desired for the output
            # quality control:
            "check_detection": False,  # if True, shows each shoreline detection to the user for validation
            "adjust_detection": False,  # if True, allows user to adjust the position of each shoreline by changing the threshold
            "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
            # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
            "min_beach_area": 4500,  # minimum area (in metres^2) for an object to be labelled as a beach
            "min_length_sl": 100,  # minimum length (in metres) of shoreline perimeter to be valid
            "cloud_mask_issue": False,  # switch this parameter to True if sand pixels are masked (in black) on many images
            "sand_color": "default",  # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
            "pan_off": "False",  # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
            "max_dist_ref": 25,
            "along_dist": 25,
            "landsat_collection": "C02",
        }
        if settings is not None:
            tmp_settings = {**settings, **self.settings}
            self.settings = tmp_settings.copy()
        # selected_set set(str): ids of the selected rois
        self.selected_set = set()
        # self.extracted_shoreline_layers : names of extracted shorelines vectors on the map
        self.extracted_shoreline_layers = []
        self.rois = None
        # self.transect : transect object containing transects loaded on map
        self.transects = None
        # self.shoreline : shoreline object containing shoreline loaded on map
        self.shoreline = None
        # Bbox saved by the user
        self.bbox = None
        # preprocess_settings : dictionary of settings used by coastsat to download imagery
        self.preprocess_settings = {}
        # create map if map_settings not provided else use default settings
        self.map = self.create_map()
        # create controls and add to map
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.map.add(self.draw_control)
        layer_control = LayersControl(position="topright")
        self.map.add(layer_control)
        hover_accordion_control = self.create_accordion_widget()
        self.map.add(hover_accordion_control)

    def create_map(self):
        """create an interactive map object using the map_settings
        Returns:
           ipyleaflet.Map: ipyleaflet interactive Map object
        """
        map_settings = {
            "center_point": (40.8630302395, -124.166267),
            "zoom": 13,
            "draw_control": False,
            "measure_control": False,
            "fullscreen_control": False,
            "attribution_control": True,
            "Layout": Layout(width="100%", height="100px"),
        }
        return Map(
            draw_control=map_settings["draw_control"],
            measure_control=map_settings["measure_control"],
            fullscreen_control=map_settings["fullscreen_control"],
            attribution_control=map_settings["attribution_control"],
            center=map_settings["center_point"],
            zoom=map_settings["zoom"],
            layout=map_settings["Layout"],
        )

    def create_accordion_widget(self):
        """creates a accordion style widget controller to hold data of
        a feature that was last hovered over by user on map.
        Returns:
           ipyleaflet.WidgetControl: an widget control for an accordion widget
        """
        html = HTML("Hover over a feature on the map")
        roi_html = HTML("Hover over a ROI on the map")
        roi_html.layout.margin = "0px 20px 20px 20px"
        html.layout.margin = "0px 20px 20px 20px"
        self.accordion = Accordion(
            children=[html, roi_html], titles=("Features Data", "ROI Data")
        )
        self.accordion.set_title(0, "Hover Data")
        self.accordion.set_title(1, "ROI Data")
        return WidgetControl(widget=self.accordion, position="topright")

    def load_configs(self, filepath: str):
        self.load_gdf_config(filepath)
        # path to config_gdf.json directory
        search_path = os.path.dirname(os.path.realpath(filepath))
        # create path config.json file in search_path directory
        config_file = common.find_config_json(search_path)
        config_path = os.path.join(search_path, config_file)
        logger.info(f"Loaded json config file from {config_path}")
        self.load_json_config(config_path)

    def load_gdf_config(self, filepath: str):
        print(f"Loading {filepath}")
        logger.info(f"Loading {filepath}")
        gdf = common.read_gpd_file(filepath)
        # Extract ROIs from gdf and create new dataframe
        roi_gdf = gdf[gdf["type"] == "roi"].copy()
        exception_handler.check_if_gdf_empty(
            roi_gdf,
            "ROIs from config",
            "No ROIs were present in the config file: {filepath}",
        )
        # drop all columns except id and geometry
        columns_to_drop = list(set(roi_gdf.columns) - set(["id", "geometry"]))
        logger.info(f"Dropping columns from ROI: {columns_to_drop}")
        roi_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"roi_gdf: {roi_gdf}")
        # Extract the shoreline from the gdf
        shoreline_gdf = gdf[gdf["type"] == "shoreline"].copy()
        # drop columns id and type
        if "slope" in shoreline_gdf.columns:
            columns_to_drop = ["id", "type", "slope"]
        else:
            columns_to_drop = ["id", "type"]
        logger.info(f"Dropping columns from shoreline: {columns_to_drop}")
        shoreline_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"shoreline_gdf: {shoreline_gdf}")
        # Extract the transects from the gdf
        transect_gdf = gdf[gdf["type"] == "transect"].copy()
        columns_to_drop = list(
            set(transect_gdf.columns) - set(["id", "slope", "geometry"])
        )
        logger.info(f"Dropping columns from transects: {columns_to_drop}")
        transect_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"transect_gdf: {transect_gdf}")
        # Extract bbox from gdf
        bbox_gdf = gdf[gdf["type"] == "bbox"].copy()
        columns_to_drop = list(set(bbox_gdf.columns) - set(["geometry"]))
        logger.info(f"Dropping columns from bbox: {columns_to_drop}")
        bbox_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"bbox_gdf: {bbox_gdf}")
        # delete the original gdf read in from config geojson file
        del gdf
        if bbox_gdf.empty:
            self.bbox = None
            logger.info("No Bounding Box was loaded on map")
            print("No Bounding Box was loaded on map")
        else:
            self.load_feature_on_map("bbox", gdf=bbox_gdf)
        # Create ROI object from roi_gdf
        exception_handler.check_if_gdf_empty(
            roi_gdf, "ROIs", "Cannot load empty ROIs onto map"
        )
        self.rois = ROI(rois_gdf=roi_gdf)
        self.load_feature_on_map("rois", gdf=roi_gdf)
        # Create Shoreline object from shoreline_gdf
        if shoreline_gdf.empty:
            self.shoreline = None
            logger.info("No shoreline was loaded on map")
            print("No shoreline was loaded on map")
        else:
            self.load_feature_on_map("shoreline", gdf=shoreline_gdf)

        # Create Transect object from transect_gdf
        if transect_gdf.empty:
            self.transects = None
            logger.info("No transects were loaded on map")
            print("No transects were loaded on map")
        else:
            self.load_feature_on_map("transects", gdf=transect_gdf)

    def download_imagery(self) -> None:
        """download_imagery  downloads selected rois as jpgs

        Raises:
            Exception: raised if settings is missing
            Exception: raised if 'dates','sat_list', and 'landsat_collection' are not in settings
            Exception: raised if no ROIs have been selected
        """
        # settings cannot be None
        exception_handler.check_if_None(self.settings, "settings")
        # settings must contain keys in subset
        subset = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(self.settings.keys()))
        exception_handler.check_if_subset(subset, superset, "settings")
        # selected_layer must contain selected ROI
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_layer(selected_layer, ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_roi_layer(selected_layer)
        logger.info(f"self.settings: {self.settings}")
        logger.info(f"selected_layer: {selected_layer}")
        # 1. Create a list of download settings for each ROI.
        # filepath: path to directory where downloaded imagery will be saved. Defaults to data directory in CoastSeg
        filepath = os.path.abspath(os.path.join(os.getcwd(), "data"))
        date_str = common.generate_datestring()
        roi_settings = common.create_roi_settings(
            self.settings, selected_layer.data, filepath, date_str
        )
        # Save download settings dictionary to instance of ROI
        self.rois.set_roi_settings(roi_settings)
        inputs_list = list(roi_settings.values())
        logger.info(f"inputs_list {inputs_list}")
        # 2. For each ROI used download settings to download imagery and save to jpg
        print("Download in process")
        # make a deep copy so settings doesn't get modified by the temp copy
        tmp_settings = copy.deepcopy(self.settings)
        # for each ROI use inputs dictionary to download imagery and save to jpg
        for inputs_for_roi in tqdm(inputs_list, desc="Downloading ROIs"):
            metadata = SDS_download.retrieve_images(inputs_for_roi)
            tmp_settings["inputs"] = inputs_for_roi
            logger.info(f"inputs: {inputs_for_roi}")
            logger.info(f"Saving to jpg. Metadata: {metadata}")
            SDS_preprocess.save_jpg(metadata, tmp_settings)
        # tmp settings is no longer needed
        del tmp_settings
        # 3.save settings used to download rois and the objects on map to config files
        self.save_config()
        logger.info("Done downloading")

    def load_json_config(self, filepath: str) -> None:
        exception_handler.check_if_None(self.rois)

        json_data = common.read_json_file(filepath)
        # replace coastseg_map.settings with settings from config file
        self.settings = json_data["settings"]
        logger.info(f"Loaded settings from file: {self.settings}")
        # replace roi_settings for each ROI with contents of json_data
        roi_settings = {}
        for roi_id in json_data["roi_ids"]:
            roi_settings[str(roi_id)] = json_data[roi_id]
        # Save input dictionaries for all ROIs
        self.rois.roi_settings = roi_settings
        logger.info(f"roi_settings: {roi_settings}")

    def save_config(self, filepath: str = None) -> None:
        """saves the configuration settings of the map into two files
            config.json and config.geojson
            Saves the inputs such as dates, landsat_collection, satellite list, and ROIs
            Saves the settings such as preprocess settings
        Args:
            file_path (str, optional): path to directory to save config files. Defaults to None.
        Raises:
            Exception: raised if self.settings is missing
            ValueError: raised if any of "dates", "sat_list", "landsat_collection" is missing from self.settings
            Exception: raised if self.rois is missing
            Exception: raised if selected_layer is missing
        """
        exception_handler.config_check_if_none(self.settings, "settings")
        # settings must contain keys in subset
        subset = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(self.settings.keys()))
        exception_handler.check_if_subset(subset, superset, "settings")
        exception_handler.config_check_if_none(self.rois, "ROIs")
        # selected_layer must contain selected ROI
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_roi_layer(selected_layer)
        logger.info(f"self.rois.roi_settings: {self.rois.roi_settings}")
        if self.rois.roi_settings == {}:
            if filepath is None:
                filepath = os.path.abspath(os.getcwd())
            roi_settings = common.create_roi_settings(
                self.settings, selected_layer.data, filepath
            )
            # Save download settings dictionary to instance of ROI
            self.rois.set_roi_settings(roi_settings)
        # create dictionary to be saved to config.json
        config_json = common.create_json_config(self.rois.roi_settings, self.settings)
        logger.info(f"config_json: {config_json} ")
        # get currently selected rois selected
        roi_ids = config_json["roi_ids"]
        selected_rois = self.get_selected_rois(roi_ids)
        logger.info(f"selected_rois: {selected_rois} ")
        shorelines_gdf = None
        transects_gdf = None
        bbox_gdf = None
        if self.shoreline is not None:
            shorelines_gdf = self.shoreline.gdf
        if self.transects is not None:
            transects_gdf = self.transects.gdf
        if self.bbox is not None:
            bbox_gdf = self.bbox.gdf
        # save all selected rois, shorelines, transects and bbox to config geodataframe
        config_gdf = common.create_config_gdf(
            selected_rois, shorelines_gdf, transects_gdf, bbox_gdf
        )
        logger.info(f"config_gdf: {config_gdf} ")
        is_downloaded = common.were_rois_downloaded(self.rois.roi_settings, roi_ids)
        if filepath is not None:
            # save to config.json
            common.config_to_file(config_json, filepath)
            # save to config_gdf.geojson
            common.config_to_file(config_gdf, filepath)
        elif filepath is None:
            # data has been downloaded before so inputs have keys 'filepath' and 'sitename'
            if is_downloaded == True:
                # write config_json file to each directory where a roi was saved
                roi_ids = config_json["roi_ids"]
                for roi_id in roi_ids:
                    sitename = str(config_json[roi_id]["sitename"])
                    filepath = os.path.abspath(
                        os.path.join(config_json[roi_id]["filepath"], sitename)
                    )
                    # save to config.json
                    common.config_to_file(config_json, filepath)
                    # save to config_gdf.geojson
                    common.config_to_file(config_gdf, filepath)
                print("Saved config files for each ROI")
            elif is_downloaded == False:
                # if data is not downloaded save to coastseg directory
                filepath = os.path.abspath(os.getcwd())
                # save to config.json
                common.config_to_file(config_json, filepath)
                # save to config_gdf.geojson
                common.config_to_file(config_gdf, filepath)

    def save_settings(self, **kwargs):
        """Saves the settings for downloading data in a dictionary
        Pass in data in the form of
        save_settings(sat_list=sat_list, dates=dates,**preprocess_settings)
        *You must use the names sat_list, landsat_collection, and dates
        """
        tmp_settings = self.settings.copy()
        for key, value in kwargs.items():
            tmp_settings[key] = value

        self.settings = tmp_settings.copy()
        del tmp_settings
        logger.info(f"Settings: {self.settings}")

    def update_transects_html(self, feature, **kwargs):
        # Modifies html of accordion when transect is hovered over
        default = "unknown"
        keys = [
            "id",
            "slope",
        ]
        # returns a dict with keys in keys and if a key does not exist in feature its value is default str
        values = common.get_default_dict(
            default=default, keys=keys, fill_dict=feature["properties"]
        )
        self.accordion.children[
            0
        ].value = """ 
        <h2>Transect</h2>
        <p>Id: {}</p>
        <p>Slope: {}</p>
        """.format(
            values["id"],
            values["slope"],
        )

    def update_extracted_shoreline_html(self, feature, **kwargs):
        # Modifies html body of accordion when extracted shoreline is hovered over
        default = "unknown"
        keys = [
            "date",
            "geoaccuracy",
            "cloud_cover",
            "satname",
        ]
        # returns a dict with keys in keys and if a key does not exist in feature its value is default str
        logger.info(f"feature: {feature}")
        values = common.get_default_dict(
            default=default, keys=keys, fill_dict=feature["properties"]
        )
        self.accordion.children[
            0
        ].value = """
        <h2>Extracted Shoreline</h2>
        <p>Date: {}</p>
        <p>Geoaccuracy: {}</p>
        <p>Cloud Cover: {}</p>
        <p>Satellite Name: {}</p>
        """.format(
            values["date"],
            values["geoaccuracy"],
            values["cloud_cover"],
            values["satname"],
        )

    def update_roi_html(self, feature, **kwargs):
        # Modifies html of accordion when roi is hovered over
        default = "unknown"
        keys = [
            "id",
        ]
        # returns a dict with keys in keys and if a key does not exist in feature its value is default str
        values = common.get_default_dict(
            default=default, keys=keys, fill_dict=feature["properties"]
        )
        logger.info(f"ROI feature: {feature}")
        roi_area = common.get_area(feature["geometry"])
        self.accordion.children[
            1
        ].value = """ 
        <h2>ROI</h2>
        <p>Id: {}</p>
        <p>Area(m²): {}</p>
        """.format(
            values["id"], roi_area
        )

    def update_shoreline_html(self, feature, **kwargs):
        # Modifies html of accordion when shoreline is hovered over
        default_str = "unknown"
        keys = [
            "MEAN_SIG_WAVEHEIGHT",
            "TIDAL_RANGE",
            "ERODIBILITY",
            "river_label",
            "sinuosity_label",
            "slope_label",
            "turbid_label",
            "CSU_ID",
        ]
        # returns a dict with keys in keys and if a key does not exist in feature its value is default str
        values = common.get_default_dict(
            default=default_str, keys=keys, fill_dict=feature["properties"]
        )
        self.accordion.children[
            0
        ].value = """
        <h2>Shoreline</h2>
        <p>Mean Sig Waveheight: {}</p>
        <p>Tidal Range: {}</p>
        <p>Erodibility: {}</p>
        <p>River: {}</p>
        <p>Sinuosity: {}</p>
        <p>Slope: {}</p>
        <p>Turbid: {}</p>
        <p>CSU_ID: {}</p>
        """.format(
            values["MEAN_SIG_WAVEHEIGHT"],
            values["TIDAL_RANGE"],
            values["ERODIBILITY"],
            values["river_label"],
            values["sinuosity_label"],
            values["slope_label"],
            values["turbid_label"],
            values["CSU_ID"],
        )

    def load_extracted_shorelines_to_map(self, roi_ids: list) -> None:
        # for each ROI that has extracted shorelines load onto map
        for roi_id in roi_ids:
            roi_extract_shoreline = self.rois.extracted_shorelines[roi_id]
            logger.info(roi_extract_shoreline)
            if roi_extract_shoreline is not None:
                self.load_extracted_shorelines_on_map(roi_extract_shoreline)

    def save_extracted_shorelines_to_file(self, roi_ids: list) -> None:
        # Saves extracted_shorelines to ROI's directory to file 'extracted_shorelines.geojson'
        for roi_id in roi_ids:
            roi_extract_shoreline = self.rois.extracted_shorelines[roi_id]
            if roi_extract_shoreline is not None:
                sitename = self.rois.roi_settings[roi_id]["sitename"]
                filepath = self.rois.roi_settings[roi_id]["filepath"]
                roi_extract_shoreline.save_to_file(sitename, filepath)

    def get_most_accurate_epsg(self, settings: dict, bbox: gpd.GeoDataFrame) -> int:
        """Returns most accurate epsg code based on lat and lon if output epsg
        was 4326 or 4327

        Args:
            settings (dict): settings for map must contain output_epsg
            bbox (gpd.GeoDataFrame): geodataframe for bounding box on map

        Returns:
            int: epsg code that is most accurate or unchanged if crs not 4326 or 4327
        """
        output_epsg = settings["output_epsg"]
        logger.info(f"Old output_epsg:{output_epsg}")
        # coastsat cannot use 4326 to extract shorelines so modify output_epsg
        if output_epsg == 4326 or output_epsg == 4327:
            geometry = bbox.iloc[0]["geometry"]
            output_epsg = common.get_epsg_from_geometry(geometry)
            logger.info(f"New output_epsg:{output_epsg}")
        return output_epsg

    def extract_all_shorelines(self) -> None:
        """Use this function when the user interactively downloads rois
        Iterates through all the ROIS downloaded by the user as indicated by the roi_settings generated by
        download_imagery() and extracts a shoreline for each of them
        """
        # ROIs,settings, roi-settings cannot be None or empty
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.shoreline, "shoreline")
        exception_handler.check_empty_dict(self.rois.roi_settings, "roi_settings")
        exception_handler.check_if_None(self.settings, "settings")
        # settings must contain keys in subset
        subset = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(self.settings.keys()))
        exception_handler.check_if_subset(subset, superset, "settings")
        # roi_settings must contain roi ids in selected set
        subset = self.selected_set
        superset = set(list(self.rois.roi_settings.keys()))
        error_message = "To extract shorelines you must first select ROIs and have the data downloaded."
        exception_handler.check_if_subset(
            subset, superset, "roi_settings", error_message
        )
        # if no rois are selected throw an error
        exception_handler.check_selected_set(self.selected_set)
        roi_ids = list(self.selected_set)
        logger.info(f"roi_ids: {roi_ids}")
        exception_handler.check_if_rois_downloaded(self.rois.roi_settings, roi_ids)

        exception_handler.check_if_None(self.bbox, "bounding box")
        # if output_epsg is 4326 or 4327 change output_epsg to most accurate crs
        self.settings["output_epsg"] = self.get_most_accurate_epsg(
            self.settings, self.bbox.gdf
        )
        # update configs with new output_epsg
        self.save_config()

        # extracted_shoreline_dict: holds extracted shorelines for each ROI
        extracted_shoreline_dict = {}
        # get selected ROIs on map and extract shoreline for each of them
        for roi_id in tqdm(roi_ids, desc="Extracting Shorelines"):
            try:
                print(f"Extracting shorelines from ROI with the id:{roi_id}")
                roi_settings = self.rois.roi_settings[roi_id]
                single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)
                # Clip shoreline to specific roi
                shoreline_in_roi = gpd.clip(self.shoreline.gdf, single_roi)
                # extract shorelines from ROI
                extracted_shorelines = extracted_shoreline.Extracted_Shoreline(
                    roi_id,
                    shoreline_in_roi,
                    roi_settings,
                    self.settings,
                )
                logger.info(
                    f"extracted_shoreline_dict[{roi_id}]: {extracted_shorelines}"
                )
            except exceptions.Id_Not_Found as id_error:
                logger.warning(f"exceptions.Id_Not_Found {id_error}")
                print(f"exceptions.Id_Not_Found:{id_error}. \n Skipping to next ROI")
            except exceptions.No_Extracted_Shoreline as no_shoreline:
                extracted_shoreline_dict[roi_id] = None
                logger.warning(f"{no_shoreline}")
                print(no_shoreline)
            else:
                extracted_shoreline_dict[roi_id] = extracted_shorelines

        # Save all the extracted_shorelines to ROI
        self.rois.update_extracted_shorelines(extracted_shoreline_dict)
        logger.info(f"rois.extracted_shorelines {self.rois.extracted_shorelines}")
        # Saves extracted_shorelines to ROI's directory to file 'extracted_shorelines.geojson'
        self.save_extracted_shorelines_to_file(roi_ids)
        # for each ROI that has extracted shorelines load onto map
        self.load_extracted_shorelines_to_map(roi_ids)

    def get_selected_rois(self, roi_ids: list) -> gpd.GeoDataFrame:
        """Returns a geodataframe of all rois selected by roi_ids

        Args:
            roi_ids (list[str]): ids of ROIs

        Returns:
            gpd.GeoDataFrame:  geodataframe of all rois selected by the roi_ids
        """
        selected_rois_gdf = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]
        return selected_rois_gdf

    def compute_transects_from_roi(
        self,
        roi_id: str,
        extracted_shorelines: list,
        transects_gdf: gpd.GeoDataFrame,
        settings: dict,
    ) -> dict:
        """Computes the intersection between the 2D shorelines and the shore-normal.
            transects. It returns time-series of cross-shore distance along each transect.
        Args:
            roi_id(str): id of roi that transects intersect
            extracted_shorelines (list): contains the extracted shorelines and corresponding metadata
            transects_gdf (gpd.GeoDataFrame): transects in ROI with crs= output_crs in settings
            settings (dict): settings dict with keys
                        'along_dist': int
                            alongshore distance considered calculate the intersection
        Returns:
            dict:  time-series of cross-shore distance along each of the transects.
                   Not tidally corrected.
        """
        # create dict of numpy arrays of transect start and end points
        transects = common.get_transect_points_dict(roi_id, transects_gdf)
        logger.info(f"transects: {transects}")
        # cross_distance: along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
        cross_distance = SDS_transects.compute_intersection(
            extracted_shorelines, transects, settings
        )
        return cross_distance

    def compute_transects(self) -> dict:
        """Returns a dict of cross distances for each roi's transects

        Args:
            selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id
                    'geometry':{
                        'type':Polygon
                        'coordinates': list of coordinates that make up polygon
                    }
                ],...}
            extracted_shorelines (dict): dictionary with roi_id keys that identify roi associates with shorelines
            {
                roi_id:{
                    dates: [datetime.datetime,datetime.datetime], ...
                    shorelines: [array(),array()]    }
            }

            settings (dict): settings used for CoastSat. Must have the following fields:
               'output_epsg': int
                    output spatial reference system as EPSG code
                'along_dist': int
                    alongshore distance considered caluclate the intersection

        Returns:
            dict: cross_distances_rois with format:
            { roi_id :  dict
                time-series of cross-shore distance along each of the transects. Not tidally corrected. }
        """
        cross_distance_transects = {}
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.transects, "transects")
        exception_handler.check_empty_dict(
            self.rois.extracted_shorelines, "extracted_shorelines"
        )
        exception_handler.check_if_None(self.settings, "settings")
        exception_handler.check_if_subset(
            set(["along_dist"]), set(list(self.settings.keys())), "settings"
        )
        # ids of ROIs that have had their shorelines extracted
        extracted_shoreline_ids = set(list(self.rois.extracted_shorelines.keys()))
        logger.info(f"extracted_shoreline_ids:{extracted_shoreline_ids}")

        # Get ROI ids that are selected on map and have had their shorelines extracted
        roi_ids = list(extracted_shoreline_ids & self.selected_set)
        exception_handler.check_if_list_empty(roi_ids)
        # user selected output projection
        output_epsg = "epsg:" + str(self.settings["output_epsg"])
        # Save cross distances for each set of transects intersecting each extracted shoreline for each ROI
        cross_distance_transects = {}
        for roi_id in tqdm(roi_ids, desc="Computing Cross Distance Transects"):
            failure_msg = ""
            cross_distance = None
            try:
                # if no extracted shoreline exist for ROI's id return cross distance = 0
                roi_extracted_shoreline = self.rois.extracted_shorelines[str(roi_id)]
                if roi_extracted_shoreline is None:
                    cross_distance = 0
                    failure_msg = f"No shorelines were extracted for {roi_id}"
                elif roi_extracted_shoreline is not None:
                    single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)
                    # get extracted shorelines array
                    extracted_shorelines = roi_extracted_shoreline.extracted_shorelines
                    transects_in_roi_gdf = gpd.sjoin(
                        left_df=self.transects.gdf,
                        right_df=single_roi,
                        how="inner",
                        predicate="intersects",
                    )
                    columns_to_drop = list(
                        set(transects_in_roi_gdf.columns) - set(["id_left", "geometry"])
                    )
                    transects_in_roi_gdf.drop(columns_to_drop, axis=1, inplace=True)
                    transects_in_roi_gdf.rename(columns={"id_left": "id"}, inplace=True)
                    if transects_in_roi_gdf.empty:
                        cross_distance = 0
                        failure_msg = f"No transects intersected {roi_id}"
                    # Check if any extracted shorelines in ROI intersect with transect
                    extracted_shoreline_x_transect = gpd.sjoin(
                        left_df=transects_in_roi_gdf,
                        right_df=roi_extracted_shoreline.gdf,
                        how="inner",
                        predicate="intersects",
                    )
                    if extracted_shoreline_x_transect.empty:
                        cross_distance = 0
                        failure_msg = f"No extracted shorelines intersected transects for {roi_id}"
                    del extracted_shoreline_x_transect
                    # convert transects_in_roi_gdf to output_crs from settings
                    transects_in_roi_gdf = transects_in_roi_gdf.to_crs(output_epsg)
                    # if shoreline and transects in ROI and extracted_shorelines intersect transect
                    # compute cross distances of transects and extracted shorelines
                    if cross_distance is None:
                        cross_distance = self.compute_transects_from_roi(
                            roi_id,
                            extracted_shorelines,
                            transects_in_roi_gdf,
                            self.settings,
                        )
                if cross_distance == 0:
                    logger.warning(failure_msg)
                    print(failure_msg)
                # save cross distances by ROI id
                cross_distance_transects[roi_id] = cross_distance
                logger.info(
                    f"\ncross_distance_transects[{roi_id}]: {cross_distance_transects[roi_id]}"
                )
                self.rois.save_transects_to_json(roi_id, cross_distance)
            except exceptions.No_Extracted_Shoreline:
                logger.warning(
                    f"ROI id:{roi_id} has no extracted shoreline. No transects computed"
                )
                print(
                    f"ROI id:{roi_id} has no extracted shoreline. No transects computed"
                )
            # save cross distances for all transects to ROIs
            self.rois.cross_distance_transects = cross_distance_transects

    def save_transects_to_csv(self) -> None:
        """save_cross_distance_df Saves the cross distances of the transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Raises:
            Exception: No ROIs have been loaded
            Exception: No transects were loaded
            Exception: No shorelines have been extracted
            Exception: No roi settings have been loaded
            Exception: No rois selected that have had shorelines extracted
        """
        # ROIs and transects must exist
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.transects, "transects")
        # there must be extracted shorelines for rois
        exception_handler.check_empty_dict(
            self.rois.extracted_shorelines, "extracted_shorelines"
        )
        exception_handler.check_empty_dict(self.rois.roi_settings, "roi_settings")
        #  each roi must have a computed transect
        exception_handler.check_empty_dict(
            self.rois.cross_distance_transects, "cross_distance_transects"
        )

        # only get roi ids that are currently selected on map and have had their shorelines extracted
        extracted_shoreline_ids = set(list(self.rois.extracted_shorelines.keys()))
        roi_ids = list(extracted_shoreline_ids & self.selected_set)
        if len(roi_ids) == 0:
            logger.error(
                f"{self.selected_set} was not a subset of extracted_shoreline_ids: {extracted_shoreline_ids}"
            )
            raise Exception(
                f"You must select an ROI and extract it shorelines before you can compute transects"
            )
        # save cross distances for transects and extracted shorelines to csv file
        # each csv file is saved to ROI directory
        self.save_cross_distance_df(roi_ids, self.rois)

    def save_cross_distance_df(self, roi_ids: list, rois: ROI) -> None:
        """Saves cross distances of transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Args:
            roi_ids (list): list of roi ids
            rois (ROI): ROI instance containing keys:
                'extracted_shorelines': extracted shoreline from roi
                'roi_settings': must have keys 'filepath' and 'sitename'
                'cross_distance_transects': cross distance of transects and extracted shoreline from roi
        """
        for roi_id in roi_ids:
            roi_extracted_shorelines = rois.extracted_shorelines[roi_id]
            # if roi does not have extracted shoreline skip it
            if roi_extracted_shorelines is None:
                print(f"ROI: {roi_id} had no extracted shorelines")
                print(
                    f"ROI: {roi_id} will not have time-series of shoreline change along transects "
                )
                logger.info(f"ROI: {roi_id} had no extracted shorelines ")
                continue
            # get extracted_shorelines from extracted shoreline object in rois
            extracted_shorelines = roi_extracted_shorelines.extracted_shorelines
            cross_distance_transects = rois.cross_distance_transects[roi_id]
            logger.info(f"ROI: {roi_id} cross distance : {cross_distance_transects}")
            logger.info(f"ROI: {roi_id} extracted_shorelines : {extracted_shorelines}")
            # if no cross distance was 0 then skip
            if cross_distance_transects == 0:
                print(f"ROI: {roi_id} cross distance is 0")
                print(
                    f"ROI: {roi_id} will not have time-series of shoreline change along transects "
                )
                logger.info(f"ROI: {roi_id} cross distance is 0")
                continue
            # if no shorelines were extracted then skip
            if extracted_shorelines is None:
                print(f"ROI: {roi_id} had no extracted shorelines")
                print(
                    f"ROI: {roi_id} will not have time-series of shoreline change along transects "
                )
                logger.info(f"ROI: {roi_id} had no extracted shorelines ")
                continue
            # check if cross_distance transects is 0
            cross_distance_df = common.get_cross_distance_df(
                extracted_shorelines, cross_distance_transects
            )
            logger.info(f"ROI: {roi_id} cross_distance_df : {cross_distance_df}")
            filepath = rois.roi_settings[roi_id]["filepath"]
            sitename = rois.roi_settings[roi_id]["sitename"]
            fn = os.path.join(filepath, sitename, "transect_time_series.csv")
            if os.path.exists(fn):
                print(f"Overwriting:{fn}")
                os.remove(fn)
            cross_distance_df.to_csv(fn, sep=",")
            print(f"ROI: {roi_id} time-series of shoreline change along transects")
            print(
                f"Time-series of the shoreline change along the transects saved as:{fn}"
            )

    def remove_all(self):
        """Remove the bbox, shoreline, all rois from the map"""
        self.remove_bbox()
        self.remove_shoreline()
        self.remove_transects()
        self.remove_all_rois()
        self.remove_accordion_html()
        self.remove_layer_by_name("geodataframe")
        self.remove_extracted_shorelines()

    def remove_bbox(self):
        """Remove all the bounding boxes from the map"""
        if self.bbox is not None:
            del self.bbox
            self.bbox = None
        self.draw_control.clear()
        existing_layer = self.map.find_layer(Bounding_Box.LAYER_NAME)
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
        self.bbox = None

    def remove_extracted_shorelines(self):
        """Remove all the extracted shorelines layers from map"""
        for layer in self.extracted_shoreline_layers:
            self.remove_layer_by_name(layer)
        self.extracted_shoreline_layers = []

    def remove_layer_by_name(self, layer_name: str):
        existing_layer = self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
        logger.info(f"Removed layer {layer_name}")

    def remove_accordion_html(self):
        """Clear the shoreline html accordion"""
        self.accordion.children[0].value = "Hover over a feature on the map."
        self.accordion.children[1].value = "Hover over a ROI on the map."

    def remove_shoreline(self):
        del self.shoreline
        self.remove_layer_by_name(Shoreline.LAYER_NAME)
        self.shoreline = None

    def remove_transects(self):
        del self.transects
        self.transects = None
        self.remove_layer_by_name(Transects.LAYER_NAME)

    def replace_layer_by_name(
        self, layer_name: str, new_layer: GeoJSON, on_hover=None, on_click=None
    ) -> None:
        """Replaces layer with layer_name with new_layer on map. Adds on_hover and on_click callable functions
        as handlers for hover and click events on new_layer
        Args:
            layer_name (str): name of layer to replace
            new_layer (GeoJSON): ipyleaflet GeoJSON layer to add to map
            on_hover (callable, optional): Callback function that will be called on hover event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
            on_click (callable, optional): Callback function that will be called on click event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
        """
        logger.info(
            f"layer_name {layer_name} \non_hover {on_hover}\n on_click {on_click}"
        )
        self.remove_layer_by_name(layer_name)
        exception_handler.check_empty_layer(new_layer, layer_name)
        # when feature is hovered over on_hover function is called
        if on_hover is not None:
            new_layer.on_hover(on_hover)
        if on_click is not None:
            # when feature is clicked on on_click function is called
            new_layer.on_click(on_click)
        self.map.add_layer(new_layer)
        logger.info(f"Add layer to map: {layer_name}")

    def remove_all_rois(self) -> None:
        """Removes all the unselected rois from the map"""
        logger.info("Removing all ROIs from map")
        # Remove the selected and unselected rois
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        self.remove_layer_by_name(ROI.LAYER_NAME)
        # clear all the ids from the selected set
        self.selected_set = set()

    def create_DrawControl(self, draw_control: "ipyleaflet.leaflet.DrawControl"):
        """Modifies given draw control so that only rectangles can be drawn

        Args:
            draw_control (ipyleaflet.leaflet.DrawControl): draw control to modify

        Returns:
            ipyleaflet.leaflet.DrawControl: modified draw control with only ability to draw rectangles
        """
        draw_control.polyline = {}
        draw_control.circlemarker = {}
        draw_control.polygon = {}
        draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "green",
                "color": "green",
                "fillOpacity": 0.1,
                "Opacity": 0.1,
            },
            "drawError": {"color": "#dd253b", "message": "Ops!"},
            "allowIntersection": False,
            "transform": True,
        }
        return draw_control

    def handle_draw(
        self, target: "ipyleaflet.leaflet.DrawControl", action: str, geo_json: dict
    ):
        """Adds or removes the bounding box  when drawn/deleted from map
        Args:
            target (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary
        """
        if (
            self.draw_control.last_action == "created"
            and self.draw_control.last_draw["geometry"]["type"] == "Polygon"
        ):
            # validate the bbox size
            geometry = self.draw_control.last_draw["geometry"]
            bbox_area = common.get_area(geometry)
            try:
                Bounding_Box.check_bbox_size(bbox_area)
            except exceptions.BboxTooLargeError as bbox_too_big:
                self.remove_bbox()
                exception_handler.handle_bbox_error(bbox_too_big)
            except exceptions.BboxTooSmallError as bbox_too_small:
                self.remove_bbox()
                exception_handler.handle_bbox_error(bbox_too_small)
            else:
                # if no exceptions occur create new bbox, remove old bbox, and load new bbox
                logger.info(f"Made it with bbox area: {bbox_area}")
                self.load_feature_on_map("bbox")

        if self.draw_control.last_action == "deleted":
            self.remove_bbox()

    def load_extracted_shorelines_on_map(self, extracted_shoreline):
        # Loads stylized extracted shorelines onto map for single roi
        layers = extracted_shoreline.get_styled_layers()
        layer_names = extracted_shoreline.get_layer_names()
        self.extracted_shoreline_layers = [
            *self.extracted_shoreline_layers,
            *layer_names,
        ]
        logger.info(f"{layers}")
        for new_layer in layers:
            new_layer.on_hover(self.update_extracted_shoreline_html)
            self.map.add_layer(new_layer)

    def load_feature_on_map(self, feature_name, file="", gdf=None, **kwargs) -> None:
        # feature name can be ['shoreline','transects','bbox','ROIs']
        # if feature name given is not one of possible features throw exception
        # create new feature based on feature passed in and using file

        # if file is passed read gdf from file
        if file != "":
            gdf = common.read_gpd_file(file)

        new_feature = self.factory.make_feature(self, feature_name, gdf, **kwargs)
        on_hover = None
        on_click = None
        if "shoreline" in feature_name.lower():
            on_hover = self.update_shoreline_html
        if "transects" in feature_name.lower():
            on_hover = self.update_transects_html
        if "rois" in feature_name.lower():
            on_hover = self.update_roi_html
            on_click = self.geojson_onclick_handler
        # load new feature on map
        layer_name = new_feature.LAYER_NAME
        self.load_on_map(new_feature, layer_name, on_hover, on_click)

    def load_on_map(
        self, feature, layer_name: str, on_hover=None, on_click=None
    ) -> None:
        """Loads feature on map

        Replaces current feature layer on map with given feature

        Raises:
            Exception: raised if feature layer is empty
        """
        # style and add the feature to the map
        new_layer = self.create_layer(feature, layer_name)
        # Replace old feature layer with new feature layer
        self.replace_layer_by_name(
            layer_name, new_layer, on_hover=on_hover, on_click=on_click
        )

    def create_layer(self, feature, layer_name: str):
        if feature.gdf.empty:
            logger.warning("Cannot add an empty geodataframe layer to the map.")
            print("Cannot add an empty geodataframe layer to the map.")
            return None
        layer_geojson = json.loads(feature.gdf.to_json())
        # convert layer to GeoJson and style it accordingly
        styled_layer = feature.style_layer(layer_geojson, layer_name)
        return styled_layer

    def generate_ROIS_fishnet(
        self, large_len: float = 7500, small_len: float = 5000
    ) -> ROI:
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        exception_handler.check_if_None(self.bbox, "bounding box")
        logger.info(f"bbox for ROIs: {self.bbox.gdf}")
        # If no shoreline exists on map then load one in
        if self.shoreline is None:
            self.load_feature_on_map("shoreline")
        logger.info(f"self.shoreline used for ROIs:{self.shoreline}")
        # create rois within the bbox that intersect shorelines
        rois = ROI(
            self.bbox.gdf,
            self.shoreline.gdf,
            square_len_lg=large_len,
            square_len_sm=small_len,
        )
        return rois

    def geojson_onclick_handler(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for when unselected geojson is clicked.

        Adds geojson's id to selected_set. Replaces current selected layer with a new one that includes
        recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        logger.info(f"properties : {properties}")
        logger.info(f"ROI_id : {properties['id']}")

        # Add id of clicked ROI to selected_set
        ROI_id = str(properties["id"])
        self.selected_set.add(ROI_id)
        logger.info(f"Added ID to selected set: {self.selected_set}")
        # remove old selected layer
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)

        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name=ROI.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        logger.info(f"selected_layer: {selected_layer}")
        self.replace_layer_by_name(
            ROI.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_onclick_handler,
            on_hover=self.update_roi_html,
        )

    def selected_onclick_handler(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for selected geojson layer.

        Removes clicked layer's cid from the selected_set and replaces the select layer with a new one with
        the clicked layer removed from select_layer.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked selected geojson. Defaults to None.
        """
        if properties is None:
            return
        # Remove the current layers cid from selected set
        logger.info(f"selected_onclick_handler: properties : {properties}")
        logger.info(f"selected_onclick_handler: ROI_id to remove : {properties['id']}")
        cid = properties["id"]
        self.selected_set.remove(cid)
        logger.info(f"selected set after ID removal: {self.selected_set}")
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        # Recreate selected layers without layer that was removed

        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name=ROI.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        self.replace_layer_by_name(
            ROI.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_onclick_handler,
            on_hover=self.update_roi_html,
        )

    def save_feature_to_file(
        self,
        feature: Union[Bounding_Box, Shoreline, Transects, ROI],
        feature_type: str = "",
    ):
        exception_handler.can_feature_save_to_file(feature, feature_type)
        if isinstance(feature, ROI):
            # raise exception if no rois were selected
            exception_handler.check_selected_set(self.selected_set)
            feature.gdf[feature.gdf["id"].isin(self.selected_set)].to_file(
                feature.filename, driver="GeoJSON"
            )
        else:
            logger.info(f"Saving feature type( {feature}) to file")
            feature.gdf.to_file(feature.filename, driver="GeoJSON")
        print(f"Save {feature.LAYER_NAME} to {feature.filename}")
        logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")

    def convert_selected_set_to_geojson(self, selected_set: set) -> dict:
        """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
        Args:
            selected_set (set): ids of selected geojson
        Returns:
           dict: geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        # create a new geojson dictionary to hold selected ROIs
        selected_rois = {"type": "FeatureCollection", "features": []}
        roi_layer = self.map.find_layer(ROI.LAYER_NAME)
        # if ROI layer does not exist throw an error
        if roi_layer is not None:
            exception_handler.check_empty_layer(roi_layer, "ROI")
        # Copy only selected ROIs with id in selected_set
        selected_rois["features"] = [
            feature
            for feature in roi_layer.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        # Each selected ROI will be blue and unselected rois will appear black
        for feature in selected_rois["features"]:
            feature["properties"]["style"] = {
                "color": "blue",
                "weight": 2,
                "fillColor": "blue",
                "fillOpacity": 0.1,
            }
        return selected_rois