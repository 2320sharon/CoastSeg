import os
from typing import Any, Collection, Dict, List, Union

import geopandas as gpd
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display
from ipywidgets import Box, VBox
from ipyleaflet import GeoJSON, Map, basemaps

"""
This code defines a FileUploader class, which provides a GUI for uploading GeoJSON files.
It uses IPython widgets to create interactive elements like dropdowns, buttons, and file choosers.
"""


class FileUploader:
    """Widget for uploading and managing GeoJSON files with interactive UI components."""

    LAYER_STYLES: Dict[str, Dict[str, Union[str, float, int]]] = {
        "shorelines": {
            "color": "#f4a261",
            "fillColor": "#f4a261",
            "fillOpacity": 0.15,
            "weight": 3,
        },
        "transects": {
            "color": "#2a9d8f",
            "fillColor": "#2a9d8f",
            "fillOpacity": 0.1,
            "weight": 3,
        },
        "shoreline extraction area": {
            "color": "#7b2cbf",
            "fillColor": "#c77dff",
            "fillOpacity": 0.18,
            "weight": 2,
        },
    }
    FEATURE_NAME_ALIASES: Dict[str, set[str]] = {
        "shorelines": {
            "shoreline",
            "shorelines",
            "reference shoreline",
            "reference shorelines",
            "reference_shoreline",
            "reference_shorelines",
        },
        "transects": {"transect", "transects"},
        "shoreline extraction area": {
            "shoreline extraction area",
            "shoreline extraction areas",
            "shoreline_extraction_area",
            "shoreline_extraction_areas",
        },
    }

    def __init__(
        self,
        title: str = "Upload a GeoJSON File",
        instructions: str = "",
        dropdown_options: Collection[str] = set(),
        filter_pattern: str = "*.geojson",
        file_selection_title: str = "",
        starting_directory: str = "",
        max_width: int = 400,
    ) -> None:
        """
        Initializes file uploader widget with customizable options.

        Args:
            title (str): Widget title text.
            instructions (str): Instructional text for users.
            dropdown_options (Set[str]): Available feature type options.
            filter_pattern (str): File filter pattern for file selection.
            file_selection_title (str): Title for file selection dialog.
            starting_directory (str): Initial directory for file chooser.
            max_width (int): Maximum widget width in pixels.
        """
        self.filenames = widgets.SelectMultiple(options=[])
        self.remove_button = widgets.Button(
            description="Remove",
            button_style="danger",
            layout=widgets.Layout(width="75px", height="28px"),
        )
        uploaded_files_title = widgets.HTML(value="<b>Uploaded Files</b>")
        self.remove_widget = widgets.VBox(
            [uploaded_files_title, widgets.HBox([self.filenames, self.remove_button])]
        )

        # Convert max_width to string form
        self.max_width = f"{max_width}px"

        # If title or instructions are empty, don't create the widget.
        self.title = widgets.HTML(value=f"<h2>{title}</h2>") if title else None
        self.instructions = (
            Box(
                [
                    widgets.HTML(
                        value=instructions,
                        layout=widgets.Layout(
                            overflow="auto",
                            white_space="pre-wrap",
                            max_width=self.max_width,
                        ),
                    )
                ],
                layout=widgets.Layout(overflow="auto", width="auto"),
            )
            if instructions
            else None
        )

        self.dropdown = widgets.Dropdown(
            options=(
                list(dropdown_options)
                if dropdown_options
                else ["transects", "shorelines", "shoreline extraction area"]
            )
        )

        self.files_dict: Dict[str, str] = {}
        self.file_dialog = FileChooser(starting_directory)
        self.file_dialog_row = widgets.HBox([self.file_dialog])
        self.preview_title = widgets.HTML(value="<b>Uploaded Feature Preview</b>")
        self.preview_status = widgets.HTML(
            value=(
                "<span style='color:#bdbdbd;'>Optional step. Skip uploads to use the "
                "shorelines and transects CoastSeg loads automatically.</span>"
            )
        )
        self.preview_map = Map(
            basemap=basemaps.CartoDB.DarkMatter,
            center=(20, 0),
            zoom=2,
            scroll_wheel_zoom=True,
            layout=widgets.Layout(width="100%", height="320px"),
        )
        self.preview_layers: Dict[str, GeoJSON] = {}
        self.preview_widget = widgets.VBox(
            [self.preview_title, self.preview_status, self.preview_map],
            layout=widgets.Layout(min_width="360px", width="100%"),
        )

        self._initialize_file_dialog(filter_pattern, file_selection_title)

        # Register event handlers
        self.remove_button.on_click(self.remove_file)

    def _initialize_file_dialog(
        self, filter_pattern: Union[str, List[str]], file_selection_title: str
    ) -> None:
        """
        Configures file dialog with filter and callback.

        Args:
            filter_pattern (Union[str, List[str]]): File pattern filter(s).
            file_selection_title (str): Title for file selection dialog.
        """
        self.file_dialog.title = f"<b>{file_selection_title}</b>"
        self.file_dialog.filter_pattern = (
            filter_pattern if isinstance(filter_pattern, list) else [filter_pattern]
        )
        self.file_dialog.register_callback(self.save_file)

    def remove_file(self, button: widgets.Button) -> None:
        """
        Removes selected files from the uploaded files list.

        Args:
            button (widgets.Button): Button widget that triggered the event.
        """
        selected_files = self.filenames.value
        for selected_file in selected_files:
            keys_to_remove = [
                feature
                for feature, path in self.files_dict.items()
                if path == selected_file
            ]
            self.files_dict = {
                k: v for k, v in self.files_dict.items() if v != selected_file
            }
            for feature in keys_to_remove:
                self._remove_preview_layer(feature)
        self.filenames.options = list(self.files_dict.values())
        self._update_preview_status()

    def save_file(self, selected: Any) -> None:
        """
        Saves selected file to the files dictionary.

        Args:
            selected (Any): Selected file information from file chooser.
        """
        feature = self.dropdown.value
        self.files_dict[feature] = self.file_dialog.selected
        self.filenames.options = list(self.files_dict.values())
        self._update_preview_layer(feature, self.file_dialog.selected)
        # Clear the file selection
        self.file_dialog.reset()

    def _update_preview_status(self) -> None:
        """Update preview help text based on uploaded files."""
        if not self.files_dict:
            self.preview_status.value = (
                "<span style='color:#bdbdbd;'>Optional step. Skip uploads to use the "
                "shorelines and transects CoastSeg loads automatically.</span>"
            )
            return

        loaded_features = ", ".join(sorted(self.files_dict.keys()))
        self.preview_status.value = (
            "<span style='color:#d9d9d9;'>Previewing uploaded feature"
            f"{'s' if len(self.files_dict) > 1 else ''}: {loaded_features}.</span>"
        )

    def _remove_preview_layer(self, feature: str) -> None:
        """Remove a preview layer from the map if it exists."""
        layer = self.preview_layers.pop(feature, None)
        if layer is not None:
            self.preview_map.remove_layer(layer)

    @staticmethod
    def _normalize_feature_name(value: Any) -> str:
        """Normalize feature type names for case-insensitive matching."""
        if value is None:
            return ""
        normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
        normalized = " ".join(normalized.split())
        if normalized.endswith("es") and normalized[:-2] in {
            "shorelin",
            "transect",
        }:
            normalized = normalized[:-2]
        elif normalized.endswith("s") and not normalized.endswith("ss"):
            singular = normalized[:-1]
            if singular in {
                "shoreline",
                "transect",
                "reference shoreline",
                "shoreline extraction area",
            }:
                normalized = singular
        return normalized

    def _filter_preview_gdf(
        self, feature: str, filepath: str, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Filter config_gdf uploads to the selected feature type; render other files unchanged."""
        if os.path.basename(filepath).lower() != "config_gdf.geojson":
            return gdf
        if "type" not in gdf.columns:
            return gdf

        valid_types = {
            self._normalize_feature_name(name)
            for name in self.FEATURE_NAME_ALIASES.get(feature, {feature})
        }
        normalized_types = gdf["type"].map(self._normalize_feature_name)
        return gdf[normalized_types.isin(valid_types)].copy()

    def _update_preview_layer(self, feature: str, filepath: str) -> None:
        """Load a GeoJSON file onto the preview map."""
        self._remove_preview_layer(feature)
        gdf = gpd.read_file(filepath)
        preview_source = self._filter_preview_gdf(feature, filepath, gdf)
        if preview_source.empty:
            self.preview_status.value = f"<span style='color:#f4a261;'>No {feature} features were found to preview in {os.path.basename(filepath)}.</span>"
            return

        preview_gdf = (
            preview_source.to_crs("EPSG:4326")
            if preview_source.crs
            else preview_source.set_crs("EPSG:4326")
        )
        geojson_layer = GeoJSON(
            data=preview_gdf.__geo_interface__,
            name=feature,
            style=self.LAYER_STYLES.get(feature, self.LAYER_STYLES["shorelines"]),
            hover_style={"weight": 4, "fillOpacity": 0.25},
        )
        self.preview_layers[feature] = geojson_layer
        self.preview_map.add_layer(geojson_layer)
        bounds = preview_gdf.total_bounds
        min_x, min_y, max_x, max_y = bounds.tolist()
        if min_x == max_x and min_y == max_y:
            self.preview_map.center = (min_y, min_x)
            self.preview_map.zoom = 12
        else:
            self.preview_map.fit_bounds(((min_y, min_x), (max_y, max_x)))
        self._update_preview_status()

    def _get_widgets_to_display(self) -> List[widgets.Widget]:
        """
        Gets ordered list of widgets for display.

        Returns:
            List[widgets.Widget]: Non-null widgets in display order.
        """
        # Order of widgets
        return [
            w
            for w in [
                self.title,
                self.instructions,
                self.dropdown,
                self.remove_widget,
                self.file_dialog_row,
            ]
            if w is not None
        ]

    def display(self) -> None:
        """Displays all file uploader widgets in the current output."""
        display(*self._get_widgets_to_display())

    def get_FileUploader_widget(self) -> VBox:
        """
        Gets file uploader widget as a VBox container.

        Returns:
            VBox: Container widget with all file uploader components.
        """
        controls = VBox(
            self._get_widgets_to_display(),
            layout=widgets.Layout(max_width=self.max_width, width="100%"),
        )
        return VBox(
            [
                widgets.HBox(
                    [controls, self.preview_widget],
                    layout=widgets.Layout(
                        width="100%",
                        align_items="flex-start",
                        flex_flow="row wrap",
                        gap="16px",
                    ),
                )
            ],
            layout=widgets.Layout(width="100%"),
        )
