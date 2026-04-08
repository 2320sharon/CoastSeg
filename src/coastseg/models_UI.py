import contextlib
import io
import logging
import os
from typing import Optional

from ipyfilechooser import FileChooser
from IPython.display import display
from ipywidgets import (
    HTML,
    Accordion,
    Button,
    Dropdown,
    FloatText,
    HBox,
    Layout,
    Output,
    Text,
    VBox,
)

from coastseg import (
    UI_elements,
    common,
    core_utilities,
    file_utilities,
    settings_UI,
    zoo_model,
)
from coastseg.tide_correction import compute_tidal_corrections
from coastseg.upload_feature_widget import FileUploader

# icons sourced from https://fontawesome.com/v4/icons/

logger = logging.getLogger(__name__)

# syle standards
MAX_HEIGHT = "max-height: 300px;"
OVERFLOW_Y = "overflow-y: auto;"
OVERFLOW_X = "overflow-x: auto;"
TEXT_ALIGN = "text-align: center;"


class UI_Models:
    # all instances of UI will share the same debug_view
    # Keep model download status separate so download messages do not mix with run output.
    model_download_view = Output(layout={"border": "1px solid black"})
    extract_shorelines_view = Output(layout={"border": "1px solid black"})
    tidal_correction_view = Output(layout={"border": "1px solid black"})

    def __init__(self) -> None:
        """
        Initialize UI_Models with default settings and widgets.
        """
        self.settings_dashboard = settings_UI.Settings_UI()
        self.zoo_model_instance = zoo_model.Zoo_Model()
        instructions = "This step is optional. Upload a GeoJSON file containing transects, shorelines, or a shoreline extraction area only if you want to override the default files CoastSeg loads automatically.\
        If no file is provided, CoastSeg will attempt to load available shorelines and transects for you.\
        If no compatible default files exist for the selected region, you will need to upload your own files."
        self.fileuploader = FileUploader(
            title="",
            instructions=instructions,
            filter_pattern="*geojson",
            dropdown_options=["transects", "shorelines", "shoreline extraction area"],
            file_selection_title="Select a geojson file",
            max_width=400,
        )
        # Controls size of ROIs generated on map
        self.model_dict = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "BEST",
            "model_type": "global_segformer_RGB_4class_14036903",
            "otsu": False,
            "tta": False,
            "img_type": "RGB",
        }
        # list of RGB and MNDWI models available
        self.RGB_models = [
            "global_segformer_RGB_4class_14036903",  # global segformer model
            "AK_segformer_RGB_4class_14037041",  # AK segformer model
        ]
        self.MNDWI_models = [
            "global_segformer_MNDWI_4class_14183366",
            "AK_segformer_MNDWI_4class_14187478 ",  # AK segformer model
        ]
        self.NDWI_models = [
            "global_segformer_NDWI_4class_14172182",  # global segformer model
            "AK_segformer_NDWI_4class_14183210",  # AK segformer model
        ]
        self.session_name = ""
        self.shoreline_session_directory = ""
        self.model_session_directory = ""
        self.shoreline_session_name = ""
        self.segmentation_session_directory = ""

        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()
        self.create_tidal_correction_widget()
        self.warning_row1 = HBox([])
        self.warning_row2 = HBox([])

    def get_warning_box(self, position: int = 1) -> HBox:
        if position == 2:
            return self.warning_row2
        return self.warning_row1  # default return warning row 1

    def clear_extract_shorelines_btn(self) -> Button:
        """
        Create button to clear extract shorelines output view.

        Returns:
            Button: Clear button widget.
        """
        clear_button = Button(
            description="Clear TextBox", style=dict(button_color="#a3adac")
        )
        clear_button.on_click(self.clear_extract_shorelines_view)
        return clear_button

    def clear_tidal_correction_btn(self) -> Button:
        """
        Create button to clear tidal correction output view.

        Returns:
            Button: Clear button widget.
        """
        clear_button = Button(
            description="Clear TextBox", style=dict(button_color="#a3adac")
        )
        clear_button.on_click(self.clear_tidal_correction_view)
        return clear_button

    def clear_extract_shorelines_view(self, btn: Button) -> None:
        """
        Clear extract shorelines output view.

        Args:
            btn (Button): Button widget that triggered the event.
        """
        UI_Models.extract_shorelines_view.clear_output()

    def clear_tidal_correction_view(self, btn: Button) -> None:
        """
        Clear tidal correction output view.

        Args:
            btn (Button): Button widget that triggered the event.
        """
        UI_Models.tidal_correction_view.clear_output()

    def get_model_instance(self) -> zoo_model.Zoo_Model:
        """
        Get the zoo model instance.

        Returns:
            zoo_model.Zoo_Model: The zoo model instance.
        """
        return self.zoo_model_instance

    def set_session_name(self, name: str) -> None:
        """
        Set the session name.

        Args:
            name (str): Session name to set.
        """
        self.session_name = str(name).strip()

    def get_session_name(self) -> str:
        """
        Get the current session name.

        Returns:
            str: Current session name.
        """
        return self.session_name

    def set_shoreline_session_name(self, name: str) -> None:
        """
        Set the shoreline session name.

        Args:
            name (str): Shoreline session name to set.
        """
        self.shoreline_session_name = str(name).strip()

    def get_shoreline_session_name(self) -> str:
        """
        Get the current shoreline session name.

        Returns:
            str: Current shoreline session name.
        """
        return self.shoreline_session_name

    def get_shoreline_session_selection(self) -> VBox:
        """
        Create widget for shoreline session name selection.

        Returns:
            VBox: Widget containing session name input and controls.
        """
        output = Output()
        self.shoreline_session_name_text = Text(
            value="",
            placeholder="Enter name",
            description="Extracted Shoreline Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = Button(description="Enter")

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            session_name = str(self.shoreline_session_name_text.value).strip()
            base_path = os.path.abspath(core_utilities.get_base_dir())
            session_path = file_utilities.create_directory(base_path, "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(
                    f"Session {session_name} already exists at {new_session_path} and will be overwritten."
                )
                self.set_shoreline_session_name(session_name)
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} will be created at {new_session_path}")
                self.set_shoreline_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.shoreline_session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def get_session_selection(self) -> VBox:
        """
        Create widget for session name selection.

        Returns:
            VBox: Widget containing session name input and controls.
        """
        output = Output()
        self.session_name_text = Text(
            value="",
            placeholder="Enter a session name",
            description="Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = Button(description="Enter")

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            session_name = str(self.session_name_text.value).strip()
            base_path = os.path.abspath(core_utilities.get_base_dir())
            session_path = file_utilities.create_directory(base_path, "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(f"Session {session_name} already exists at {new_session_path}")
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} will be created at {new_session_path}")
                self.set_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def get_adv_model_settings_section(self) -> VBox:
        """
        Create advanced model settings section widget.

        Returns:
            VBox: Widget containing advanced model settings.
        """
        # declare settings widgets
        settings = {
            "model_implementation": self.model_implementation,
            "otsu": self.otsu_radio,
            "tta": self.tta_radio,
        }
        return VBox([widget for widget_name, widget in settings.items()])

    def get_basic_model_settings_section(self) -> VBox:
        """
        Create basic model settings section widget.

        Returns:
            VBox: Widget containing basic model settings.
        """
        # declare settings widgets
        settings = {
            "model_input": self.model_input_dropdown,
            "model_type": self.model_dropdown,
            "download_model": self.download_model_button,
        }
        # create settings vbox
        return VBox([widget for widget_name, widget in settings.items()])

    def get_model_settings_accordion(self) -> Accordion:
        """
        Create accordion widget containing model settings sections.

        Returns:
            Accordion: Widget with basic and advanced model settings sections.
        """
        # create settings accordion widget
        settings_accordion = Accordion(
            children=[
                self.get_basic_model_settings_section(),
                self.get_adv_model_settings_section(),
            ]
        )
        settings_accordion.set_title(0, "Basic Model Settings")
        settings_accordion.set_title(1, "Advanced Model Settings")
        settings_accordion.selected_index = 0

        return settings_accordion

    def create_dashboard(self) -> None:
        """
        Create and display the main UI dashboard with all widgets and steps.
        """
        self.file_row = HBox([])
        self.segmentation_file_row = HBox([])
        self.extracted_shoreline_file_row = HBox([])
        self.tidal_correct_file_row = HBox([])
        layout = Layout(max_width="270px", overflow="auto", padding="0px 10px 0px 0px")
        # step 1: Select Settings
        step_1 = HBox(
            [
                VBox(
                    [
                        self.step_1_instr,
                        self.save_settings_btn,
                    ],
                    layout=layout,
                ),
                self.settings_dashboard.render(),
                self.get_view_settings_vbox(),
            ]
        )
        # step 2: Upload Files
        step_2 = HBox(
            [
                HBox([self.step_2_instr], layout=layout),
                self.fileuploader.get_FileUploader_widget(),
            ]
        )

        run_model_workflow = VBox(
            [
                self.run_model_workflow_instr,
                VBox(
                    [
                        self.get_model_settings_accordion(),
                        # Show only the final download result directly under model selection.
                        UI_Models.model_download_view,
                    ],
                ),
                self.get_session_selection(),
                self.use_select_images_button,
                self.file_row,
                self.extract_shorelines_button,
            ],
            layout=Layout(
                border="2px solid #69add1",
                padding="14px",
                margin="0px 12px 0px 0px",
                min_width="355px",
                flex="1 1 355px",
                align_items="flex-start",
            ),
        )

        extract_from_folder_workflow = VBox(
            [
                self.extract_from_folder_workflow_instr,
                self.use_select_segmentation_session_button,
                self.segmentation_file_row,
                self.extract_from_segmentation_button,
            ],
            layout=Layout(
                border="2px solid #c061ff",
                padding="14px",
                margin="0px",
                min_width="355px",
                flex="1 1 355px",
                align_items="flex-start",
            ),
        )

        # step 3 : Choose Workflow
        step_3 = VBox(
            [
                self.step_3_instr,
                HBox(
                    [run_model_workflow, extract_from_folder_workflow],
                    layout=Layout(flex_flow="row wrap", align_items="stretch"),
                ),
            ]
        )

        # step 5 : Tidal Correction
        # Has the widgets: instructions, select session button,place to select session, and tidal correction widget
        step_5 = VBox(
            [
                self.step_5_instr,
                self.select_extracted_shorelines_session_button,
                self.tidal_correct_file_row,
                self.create_tidal_correction_widget(),
            ]
        )

        display(
            step_1,
            step_2,
            step_3,
            HBox(
                [self.clear_extract_shorelines_btn(), UI_Models.extract_shorelines_view]
            ),
            self.warning_row1,
            self.line_widget,
            step_5,
            HBox([self.clear_tidal_correction_btn(), UI_Models.tidal_correction_view]),
            self.warning_row2,
        )

    def create_tidal_correction_widget(self) -> VBox:
        """
        Create tidal correction widget with all necessary controls.

        Returns:
            VBox: Widget containing tidal correction controls.
        """
        load_style = dict(button_color="#69add1", description_width="initial")

        correct_tides_html = HTML(
            value="<h3><b>Apply Tide Correction</b></h3> \
               Only apply after extracting shorelines</br>",
            layout=Layout(margin="0px 5px 0px 0px"),
        )

        self.instructions_ref_elv = HTML(
            value="Refence Elevation(m) relative to user-specified vertical datum)",
            style={"description_width": "initial"},
            layout=Layout(
                width="auto",  # allows the width to adjust automatically
                min_width="100px",  # sets a minimum width
                flex="1 1 auto",  # makes it flexible within a flex container
            ),
        )
        self.reference_elevation_text = FloatText(
            value=0.0,
            description="Reference Elevation:",
            style={"description_width": "initial"},
        )
        self.beach_slope_selector = UI_elements.BeachSlopeSelector()
        self.tide_selector = UI_elements.TidesSelector()

        self.tidally_correct_button = Button(
            description="Correct Tides",
            style=load_style,
            icon="fa-tint",
        )
        self.tidally_correct_button.on_click(self.tidally_correct_button_clicked)

        return VBox(
            [
                correct_tides_html,
                self.instructions_ref_elv,
                self.reference_elevation_text,
                self.beach_slope_selector,
                self.tide_selector,
                self.tidally_correct_button,
            ]
        )

    @tidal_correction_view.capture(clear_output=True)
    def tidally_correct_button_clicked(self, button: Button) -> None:
        """
        Handle tidal correction button click event.

        Args:
            button (Button): Button widget that triggered the event.
        """
        # user must have selected imagery first
        # must select a directory of model outputs
        if self.shoreline_session_directory == "":
            self.launch_error_box(
                "Cannot correct tides",
                "Must click select session first",
                position=2,
            )
            return
        # get session directory location
        session_directory = self.shoreline_session_directory
        session_name = os.path.basename(session_directory)
        logger.info(f"session_directory: {session_directory}")
        # get roi_id
        # @todo find a better way to load the roi id
        config_json_location = file_utilities.find_file_recursively(
            session_directory, "config.json"
        )
        config = file_utilities.load_data_from_json(config_json_location)
        roi_id = config.get("roi_id", "")
        logger.info(f"roi_id: {roi_id}")

        beach_slope = self.beach_slope_selector.value
        reference_elevation = self.reference_elevation_text.value
        model = self.tide_selector.model
        tides_file = self.tide_selector.tides_file
        # load in shoreline settings, session directory with model outputs, and a new session name to store extracted shorelines
        compute_tidal_corrections(
            session_name,
            [roi_id],
            beach_slope,
            reference_elevation,
            model=model,
            tides_file=tides_file,
        )

    def _create_widgets(self) -> None:
        """
        Create dropdown widgets for model configuration.
        """
        self.model_implementation = Dropdown(
            options=["ENSEMBLE", "BEST"],
            value="BEST",
            description="Select:",
            disabled=False,
        )
        self.model_implementation.observe(self.handle_model_implementation, "value")

        self.otsu_radio = Dropdown(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Otsu Threshold:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.otsu_radio.observe(self.handle_otsu, "value")

        self.tta_radio = Dropdown(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Test Time Augmentation:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.tta_radio.observe(self.handle_tta, "value")
        self.model_input_dropdown = Dropdown(
            options=[
                "RGB",
                "MNDWI",
                "NDWI",
            ],
            value="RGB",
            description="Model Input:",
            disabled=False,
        )
        self.model_input_dropdown.observe(self.handle_model_input_change, names="value")

        self.model_dropdown = Dropdown(
            options=self.RGB_models,
            value=self.RGB_models[0],
            description="Select Model:",
            disabled=False,
        )
        self.model_dropdown.observe(self.handle_model_type, "value")

    def _create_buttons(self) -> None:
        """
        Create action buttons for the UI.
        """
        # button styles
        load_style = dict(button_color="#69add1", description_width="initial")
        action_style = dict(button_color="#ae3cf0")

        # Save settings button: saves both the model and extract shoreline settings
        self.save_settings_btn = Button(
            description="Save Settings",
            style=action_style,
            icon="floppy-o",
        )
        self.save_settings_btn.on_click(self.save_settings_clicked)

        # Extract Shorelines button: extracts shorelines from model outputs
        self.extract_shorelines_button = Button(
            description="Run Model + Extract Shorelines",
            style=dict(button_color="#0b7285"),
            icon="fa-bolt",
            layout=Layout(width="320px", min_width="320px"),
            tooltip="Run the model on selected imagery, then extract shorelines.",
        )
        self.extract_shorelines_button.on_click(self.run_model_button_clicked)

        self.extract_from_segmentation_button = Button(
            description="Extract Shorelines From Folder",
            style=dict(button_color="#8e44ad"),
            icon="fa-folder-open",
            layout=Layout(width="320px", min_width="320px"),
            tooltip="Skip model execution and extract shorelines from an existing segmentation folder.",
        )
        self.extract_from_segmentation_button.on_click(
            self.extract_from_segmentation_button_clicked
        )

        # Select Images button: selects a directory of images to run the model on
        self.use_select_images_button = Button(
            description="Select Images",
            style=load_style,
            icon="fa-file-image-o",
            layout=Layout(width="320px", min_width="320px"),
            tooltip="Choose the imagery directory used for the run-model workflow.",
        )
        self.use_select_images_button.on_click(self.use_select_images_button_clicked)

        self.use_select_segmentation_session_button = Button(
            description="Select Segmentation Folder",
            style=load_style,
            icon="fa-folder-open",
            layout=Layout(width="320px", min_width="320px"),
            tooltip="Choose an existing segmentation output folder for extraction-only workflow.",
        )
        self.use_select_segmentation_session_button.on_click(
            self.use_select_segmentation_session_button_clicked
        )

        self.download_model_button = Button(
            description="Download Model",
            style=load_style,
            icon="download",
        )
        self.download_model_button.on_click(self.download_model_button_clicked)

        self.select_extracted_shorelines_session_button = Button(
            description="Select Session",
            style=load_style,
        )
        self.select_extracted_shorelines_session_button.on_click(
            self.select_extracted_shorelines_button_clicked
        )

    def _create_HTML_widgets(self) -> None:
        """
        Create HTML widgets for displaying instructions and step information.
        """
        self.line_widget = HTML(
            value="____________________________________________________"
        )
        # step 1: Select Settings
        self.step_1_instr = HTML(
            value="<h2>Step 1: Select Settings</h2>\
            <b>1.</b> Adjust the settings used to extract shorelines from the satellite imagery\
            <br><b>2.</b> Click save to save the settings\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )
        # step 2: Upload Files
        self.step_2_instr = HTML(
            value="<h2>Step 2: Upload Files</h2>\
            <b>This step is optional.</b> Skip it if you want CoastSeg to use the automatically loaded shorelines and transects.\
            <br>If both the transects and shorelines are in the same 'config_gdf.geojson' you will need to upload it for both reference shoreline and transects.\
            <br>1. <b>Upload a Reference Shoreline (Optional): </b> Upload a GeoJSON file containing a reference shoreline.\
            <br>2. <b>Upload Transects (Optional): </b> Upload a GeoJSON file containing transects.\
            <br>3. <b>Preview uploaded files:</b> The map on the right will show uploaded shorelines, transects, or shoreline extraction areas.",
            layout=Layout(margin="0px 0px 0px 0px"),
        )
        # step 3 : Choose Workflow
        self.step_3_instr = HTML(
            value="<h2>Step 3: Choose One Workflow</h2>\
            <b>Pick exactly one option below.</b> The left workflow runs a model on imagery, while the right workflow reuses an existing segmentation folder. If you switch workflows, CoastSeg removes the previous folder selection from the other workflow so only one source is active at a time.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )
        self.run_model_workflow_instr = HTML(
            value="<h3 style='margin:0 0 8px 0;color:#69add1;'>Workflow A: Run Model Then Extract</h3>\
            <b>Use this when you have imagery only.</b>\
            <br><b>1. Select a model:</b> Choose the model input, model, and optional advanced model settings. Download the model if needed.\
            <br><b>2. Name the session:</b> Enter a session name. This becomes the name of a folder in CoastSeg/sessions that will contain the model outputs and extracted shorelines.\
            <br><b>3. Choose imagery:</b> Select the RGB directory from CoastSeg/data for an ROI that was already downloaded.\
            <br><b>4. Run processing:</b> Click 'Run Model + Extract Shorelines' to segment the imagery and extract shorelines in one workflow.\
            <br><span style='color:#f4a261;'><b>Warning:</b> This workflow only works if the Pixi environment for CoastSeg/segmentation_workflow is installed and ready to run.</span>",
            layout=Layout(margin="0px 0px 10px 0px"),
        )

        self.extract_from_folder_workflow_instr = HTML(
            value="<h3 style='margin:0 0 8px 0;color:#c061ff;'>Workflow B: Extract From Existing Segmentations</h3>\
            <b>Use this when segmentation outputs already exist.</b> Choose the folder containing segmentation outputs, model_info.json, and model_settings.json, then extract shorelines without running the model again.",
            layout=Layout(margin="0px 0px 10px 0px"),
        )
        # step 5 : Tidal Correction
        self.step_5_instr = HTML(
            value="<h2>Step 4 : Tidal Correction</h2>\
            - The tide model must have been downloaded to CoastSeg/tide_model for tidal correction to work. Follow the guide to download the tide model: https://github.com/SatelliteShorelines/CoastSeg/wiki/09.-How-to-Download-Tide-Model \
            - Ensure shorelines are extracted prior to tidal correction. Not all imagery will contain extractable shorelines, thus, tidal correction may not be possible.\
            <br><b>1. Select a Session: </b> Choose a session from the 'sessions' directory containing extracted shorelines.\
            <br><b>2. Run Tidal Correction:</b> Runs the tide model and save tidally corrected CSV files in the selected session directory.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.run_model_instr = HTML(
            value="<h2>Run a Model</h2>\
            <b>1. Session Name:</b> Enter a name.This creates a folder in the 'sessions' directory for model outputs.\
            <br><b>2. Select Images:</b> Choose an ROI directory with downloaded imagery from the 'data' directory.\
            <br><b>3. Run Model:</b> Click this button to apply the model on imagery. Extract Shorelines will be saved in the 'sessions' directory under the session name entered.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.extract_shorelines_instr = HTML(
            value="<h2>Extract Shorelines</h2>\
            Make sure to run the model before extracting shorelines.<br> \
            <b>1. Name Your Session:</b> Enter a name for your shoreline extraction session this will create a new folder in the 'sessions' directory.<br>\
            <b>2. Upload a File (Optional): </b> Upload a transects or a reference shoreline geojson file to extract shorelines with.<br>\
            <b>3. Choose Model Session</b> Select a directory from the 'sessions' directory which contains model outputs.<br>\
            <b>4. Click Extract Shorelines</b> Your extracted shorelines will be saved in the 'sessions' directory under your named session.<br><br>\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.tidally_correct_instr = HTML(
            value="<h2>Tidally Correct</h2>\
            Ensure shorelines are extracted prior to tidal correction. Not all imagery will contain extractable shorelines, thus, tidal correction may not be possible.\
            <br><b>1. Select a Session </b> Choose a session from the 'sessions' directory containing extracted shorelines.\
            <br><b>2. Correct Tides Button</b> Click to save tidally corrected CSV files in the selected session directory.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

    def handle_model_implementation(self, change: dict) -> None:
        """
        Handle model implementation dropdown change.

        Args:
            change (dict): Change event dictionary with 'new' value.
        """
        self.model_dict["implementation"] = change["new"]

    def handle_model_type(self, change: dict) -> None:
        """
        Handle model type dropdown change and manage Otsu threshold availability.

        Args:
            change (dict): Change event dictionary with 'new' value.
        """
        # 2 class model has not been selected disable otsu threhold
        self.model_dict["model_type"] = change["new"]
        if "2class" not in change["new"]:
            if self.otsu_radio.value == "Enabled":
                self.model_dict["otsu"] = False
                self.otsu_radio.value = "Disabled"
            self.otsu_radio.disabled = True
        # 2 class model was selected enable otsu threhold radio button
        if "2class" in change["new"]:
            self.otsu_radio.disabled = False

    def handle_otsu(self, change: dict) -> None:
        """
        Handle Otsu threshold dropdown change.

        Args:
            change (dict): Change event dictionary with 'new' value.
        """
        if change["new"] == "Enabled":
            self.model_dict["otsu"] = True
        if change["new"] == "Disabled":
            self.model_dict["otsu"] = False

    def handle_tta(self, change: dict) -> None:
        """
        Handle Test Time Augmentation dropdown change.

        Args:
            change (dict): Change event dictionary with 'new' value.
        """
        if change["new"] == "Enabled":
            self.model_dict["tta"] = True
        if change["new"] == "Disabled":
            self.model_dict["tta"] = False

    def handle_model_input_change(self, change: dict) -> None:
        """
        Handle model input type change and update available models.

        Args:
            change (dict): Change event dictionary with 'new' value.
        """
        if change["new"] == "RGB":
            self.model_dropdown.options = self.RGB_models
        if change["new"] == "MNDWI":
            self.model_dropdown.options = self.MNDWI_models
        if change["new"] == "NDWI":
            self.model_dropdown.options = self.NDWI_models

        self.model_dict["img_type"] = change["new"]

    @model_download_view.capture(clear_output=True)
    def download_model_button_clicked(self, button: Button) -> None:
        """Download the currently selected model and report the final status."""
        model_name_selected = str(self.model_dict.get("model_type", "")).strip()
        implementation = str(self.model_dict.get("implementation", "BEST")).upper()
        if not model_name_selected:
            self.launch_error_box(
                "Cannot Download Model",
                "Select a model before downloading.",
                position=1,
            )
            return

        button.disabled = True
        try:
            # Suppress downloader chatter so the notebook only shows the final status line.
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    model_location, downloaded = (
                        self.zoo_model_instance.ensure_model_downloaded(
                            implementation, model_name_selected
                        )
                    )
        except Exception as error:
            self.launch_error_box(
                "Model Download Failed",
                f"An error occurred while downloading the model: {error}",
                position=1,
            )
            return
        finally:
            button.disabled = False

        UI_Models.model_download_view.clear_output(wait=True)
        if downloaded:
            print(
                f"The {model_name_selected} was downloaded and saved to {model_location}"
            )
        else:
            print(f"The {model_name_selected} already exists at {model_location}")

    @extract_shorelines_view.capture(clear_output=True)
    def run_model_button_clicked(self, button: Button) -> None:
        """
        Handle extract shorelines button click event.

        Args:
            button (Button): Button widget that triggered the event.
        """
        # user must have selected imagery first
        if self.model_dict["sample_direc"] is None:
            self.launch_error_box(
                "Cannot Run Model",
                "You must click 'Select Images' first",
                position=1,
            )
            return
        # user must have selected imagery first
        session_name = self.get_session_name()
        if session_name == "":
            self.launch_error_box(
                "Cannot Run Model",
                "Must enter a session name first",
                position=1,
            )
            return
        if self.model_dict["sample_direc"] == "":
            self.launch_error_box(
                "Cannot Run Model",
                "Must click select images first and select a directory of jpgs.\n Example : C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
                position=1,
            )
            return

        if self.segmentation_session_directory:
            self.clear_selected_segmentation_directory()
            print(
                "Run-model mode was selected, so the previously selected segmentation folder was cleared before processing."
            )

        print("Running the model. Please wait.")
        zoo_model_instance = self.get_model_instance()

        # get the transects and shorelines file paths that were uploaded
        transects_path = self.fileuploader.files_dict.get("transects", "")
        shoreline_path = self.fileuploader.files_dict.get("shorelines", "")
        shoreline_extraction_area_path = self.fileuploader.files_dict.get(
            "shoreline extraction area", ""
        )
        zoo_model_instance.run_model_and_extract_shorelines(
            self.model_dict["sample_direc"],
            session_name=session_name,
            shoreline_path=shoreline_path,
            transects_path=transects_path,
            shoreline_extraction_area_path=shoreline_extraction_area_path,
        )

    def clear_selected_image_directory(self) -> None:
        """Clear the currently selected imagery directory."""
        self.model_dict["sample_direc"] = None
        if hasattr(self, "file_row"):
            common.clear_row(self.file_row)

    def clear_selected_segmentation_directory(self) -> None:
        """Clear the currently selected segmentation session directory."""
        self.segmentation_session_directory = ""
        if hasattr(self, "segmentation_file_row"):
            common.clear_row(self.segmentation_file_row)

    def get_selected_directory_from_row(self, row: HBox) -> str:
        """Return the currently selected directory from a chooser row, if any."""
        if not row.children:
            return ""

        chooser_container = row.children[0]
        if not hasattr(chooser_container, "children") or not chooser_container.children:
            return ""

        chooser = chooser_container.children[0]
        if isinstance(chooser, FileChooser) and chooser.selected:
            return os.path.abspath(chooser.selected)
        return ""

    def resolve_segmentation_session_directory(self) -> str:
        """Resolve the segmentation session directory from saved state or active chooser."""
        if self.segmentation_session_directory:
            return self.segmentation_session_directory

        session_path = self.get_selected_directory_from_row(self.segmentation_file_row)
        if not session_path:
            return ""

        self.load_segmentation_session_settings(session_path)
        self.clear_selected_image_directory()
        self.segmentation_session_directory = session_path
        print(
            "Existing-segmentation mode selected. Any previously selected image directory was cleared."
        )
        print(
            "Shorelines will be extracted from this folder using the model settings saved in that folder:\n"
            f"{session_path}"
        )
        return session_path

    def launch_segmentation_folder_error(self, error: Exception) -> None:
        """Show a specific warning for segmentation-folder validation errors."""
        error_message = str(error)
        if isinstance(error, FileNotFoundError) and (
            "model_settings.json" in error_message or "model_info.json" in error_message
        ):
            self.launch_error_box(
                "Missing Segmentation Metadata",
                error_message,
                position=1,
            )
            return

        self.launch_error_box(
            "Invalid Segmentation Folder",
            f"Select a folder that contains the segmentation outputs, model_info.json, and model_settings.json. {error}",
            position=1,
        )

    def get_segmentation_metadata_paths(self, session_path: str) -> tuple[str, str]:
        """Return required metadata file paths for a segmentation session."""
        missing_files: list[str] = []

        try:
            model_settings_path = file_utilities.find_file_by_regex(
                session_path, r"^model_settings\.json$"
            )
        except FileNotFoundError:
            model_settings_path = ""
            missing_files.append("model_settings.json")

        try:
            model_info_path = file_utilities.find_file_by_regex(
                session_path, r"^model_info\.json$"
            )
        except FileNotFoundError:
            model_info_path = ""
            missing_files.append("model_info.json")

        if missing_files:
            if len(missing_files) == 2:
                raise FileNotFoundError(
                    "The selected folder is missing required segmentation metadata files: model_settings.json and model_info.json. Select a segmentation output folder that contains both files"
                )
            raise FileNotFoundError(
                f"The selected folder is missing required segmentation metadata file: {missing_files[0]}. Select a segmentation output folder that contains this file."
            )

        return model_settings_path, model_info_path

    def load_segmentation_session_settings(self, session_path: str) -> None:
        """Load model settings from an existing segmentation session folder."""
        model_settings_path, _ = self.get_segmentation_metadata_paths(session_path)
        model_settings = file_utilities.read_json_file(
            model_settings_path, raise_error=True
        )
        if not model_settings:
            raise ValueError(
                f"model_settings.json was empty in the segmentation folder: {session_path}"
            )
        self.zoo_model_instance.set_settings(**model_settings)
        self.update_displayed_settings()

    @extract_shorelines_view.capture(clear_output=True)
    def select_RGB_callback(self, filechooser: FileChooser) -> None:
        """
        Handle directory selection for RGB images.

        Args:
            filechooser (FileChooser): File chooser widget with selected directory.
        """
        from pathlib import Path

        if filechooser.selected:
            sample_direc = Path(filechooser.selected).resolve()
            print(f"The images in the folder will be segmented :\n{sample_direc} ")
            # Using itertools.chain to combine the results of two glob calls
            has_jpgs = any(sample_direc.glob("*.jpg")) or any(
                sample_direc.glob("*.jpeg")
            )
            if not has_jpgs:
                self.launch_error_box(
                    "No jpgs found",
                    f"The directory {sample_direc} contains no jpgs! Please select a directory with jpgs. Make sure to select 'RGB' folder located in the 'preprocessed' folder.'",
                    position=1,
                )
                self.model_dict["sample_direc"] = ""
            else:
                self.clear_selected_segmentation_directory()
                self.model_dict["sample_direc"] = str(sample_direc)
                self.zoo_model_instance.set_settings(sample_direc=str(sample_direc))
                self.save_updated_settings()
                print(
                    "Run-model mode selected. Any previously selected segmentation folder was cleared."
                )

    @extract_shorelines_view.capture(clear_output=True)
    def use_select_images_button_clicked(self, button: Button) -> None:
        """
        Handle select images button click event.

        Args:
            button (Button): Button widget that triggered the event.
        """
        # Prompt the user to select a directory of images
        file_chooser = common.create_dir_chooser(
            self.select_RGB_callback, title="Select directory of images"
        )
        # clear row and close all widgets in self.file_row before adding new file_chooser
        common.clear_row(self.file_row)
        # add instance of file_chooser to self.file_row
        self.file_row.children = [file_chooser]

    @extract_shorelines_view.capture(clear_output=True)
    def select_segmentation_session_callback(self, filechooser: FileChooser) -> None:
        """Handle directory selection for extracting shorelines from segmentations."""
        if not filechooser.selected:
            return

        session_path = os.path.abspath(filechooser.selected)
        try:
            self.load_segmentation_session_settings(session_path)
        except Exception as error:
            self.segmentation_session_directory = ""
            self.launch_segmentation_folder_error(error)
            return

        self.clear_selected_image_directory()
        self.segmentation_session_directory = session_path
        print(
            "Existing-segmentation mode selected. Any previously selected image directory was cleared."
        )
        print(
            "Shorelines will be extracted from this folder using the model settings saved in that folder:\n"
            f"{session_path}"
        )

    @extract_shorelines_view.capture(clear_output=True)
    def use_select_segmentation_session_button_clicked(self, button: Button) -> None:
        """Prompt for a segmentation session directory."""
        file_chooser = common.create_dir_chooser(
            self.select_segmentation_session_callback,
            title="Select folder of segmentation outputs",
            starting_directory="sessions",
        )
        common.clear_row(self.segmentation_file_row)
        self.segmentation_file_row.children = [file_chooser]

    @extract_shorelines_view.capture(clear_output=True)
    def extract_from_segmentation_button_clicked(self, button: Button) -> None:
        """Extract shorelines from a previously generated segmentation folder."""
        try:
            session_path = self.resolve_segmentation_session_directory()
        except Exception as error:
            self.segmentation_session_directory = ""
            self.launch_segmentation_folder_error(error)
            return

        if not session_path:
            self.launch_error_box(
                "Cannot Extract From Folder",
                "You must click 'Select Segmentation Folder' first.",
                position=1,
            )
            return

        if self.model_dict["sample_direc"]:
            self.clear_selected_image_directory()
            print(
                "A segmentation folder was selected, so the previously selected image directory was cleared before extraction."
            )

        print(
            "Extracting shorelines from the selected segmentation folder. Please wait."
        )
        transects_path = self.fileuploader.files_dict.get("transects", "")
        shoreline_path = self.fileuploader.files_dict.get("shorelines", "")
        shoreline_extraction_area_path = self.fileuploader.files_dict.get(
            "shoreline extraction area", ""
        )
        self.zoo_model_instance.extract_shorelines(
            session_path=session_path,
            shoreline_path=shoreline_path,
            transects_path=transects_path,
            shoreline_extraction_area_path=shoreline_extraction_area_path,
        )

    @tidal_correction_view.capture(clear_output=True)
    def selected_shoreline_session_callback(self, filechooser: FileChooser) -> None:
        """
        Handle shoreline session directory selection.

        Args:
            filechooser (FileChooser): File chooser widget with selected directory.
        """
        if filechooser.selected:
            self.shoreline_session_directory = os.path.abspath(filechooser.selected)

    @tidal_correction_view.capture(clear_output=True)
    def select_extracted_shorelines_button_clicked(self, button: Button) -> None:
        """
        Handle select extracted shorelines session button click event.

        Args:
            button (Button): Button widget that triggered the event.
        """
        # Prompt the user to select a directory of extracted shorelines session
        file_chooser = common.create_dir_chooser(
            self.selected_shoreline_session_callback,
            title="Select extracted shorelines session",
            starting_directory="sessions",
        )
        print(f"filechooser: {file_chooser}")
        # clear row and close all widgets in self.tidal_correct_file_row before adding new file_chooser
        common.clear_row(self.tidal_correct_file_row)
        # add instance of file_chooser to self.tidal_correct_file_row
        self.tidal_correct_file_row.children = [file_chooser]

    def launch_error_box(
        self, title: Optional[str] = None, msg: Optional[str] = None, position: int = 1
    ):
        # Show user error message
        warning_box = common.create_warning_box(
            title=title, msg=msg, instructions=None, msg_width="95%", box_width="30%"
        )
        # clear row and close all widgets before adding new warning_box
        common.clear_row(self.get_warning_box(position))
        # add instance of warning_box to warning_row
        self.get_warning_box(position).children = [warning_box]

    def save_updated_settings(
        self,
    ):
        # get the settings from the settings dashboard
        settings = self.settings_dashboard.get_settings()
        # get the model settings
        settings.update(self.model_dict)
        # save the settings to the model instance
        self.zoo_model_instance.set_settings(**settings)
        # update the settings in the view settings section
        self.update_displayed_settings()

    @extract_shorelines_view.capture(clear_output=True)
    def save_settings_clicked(self, btn: Button) -> None:
        """
        Handle save settings button click event.

        Args:
            btn (Button): Button widget that triggered the event.
        """
        try:
            # get the settings from the settings dashboard
            settings = self.settings_dashboard.get_settings()
            # get the model settings
            settings.update(self.model_dict)
            # save the settings to the model instance
            self.zoo_model_instance.set_settings(**settings)
            # update the settings in the view settings section
            self.update_displayed_settings()

        except Exception as error:
            self.launch_error_box(
                "Error saving settings",
                f"An error occurred while saving settings: {error}",
                position=1,
            )

    def get_view_settings_vbox(self) -> VBox:
        # update settings button
        action_style = dict(button_color="#ae3cf0")
        refresh_settings_btn = Button(
            description="Refresh Settings", icon="refresh", style=action_style
        )
        refresh_settings_btn.on_click(self.refresh_settings_btn_clicked)
        # get the settings from the model instance
        settings = self.zoo_model_instance.get_settings()
        setting_content = format_as_html(settings)
        self.settings_html = HTML(
            f"<div style='max-height: 300px;max-width: 280px; overflow-x: auto; overflow-y:  auto; text-align: left;'>"
            f"{setting_content}"
            f"</div>"
        )
        # The view settings section: contains the settings and a button to refresh the settings
        view_settings_vbox = VBox([self.settings_html, refresh_settings_btn])
        html_settings_accordion = Accordion(
            children=[view_settings_vbox], selected_index=0
        )
        html_settings_accordion.set_title(0, "View Settings")
        return html_settings_accordion

    @extract_shorelines_view.capture(clear_output=True)
    def refresh_settings_btn_clicked(self, btn: Button) -> None:
        """
        Handle refresh settings button click event.

        Args:
            btn (Button): Button widget that triggered the event.
        """
        UI_Models.extract_shorelines_view.clear_output(wait=True)
        # refresh settings in view settings section
        try:
            self.update_displayed_settings()
        except Exception as error:
            self.launch_error_box(
                "Error saving settings",
                f"An error occurred while saving settings: {error}",
                position=1,
            )

    def update_displayed_settings(self):
        """
        Updates the displayed settings in the UI.

        Retrieves the settings from the `zoo_model_instance` and formats them as HTML.
        The formatted settings are then assigned to the `settings_html` widget.

        Returns:
            None
        """
        setting_content = format_as_html(self.zoo_model_instance.get_settings())
        self.settings_html.value = f"""<div style='max-height: 300px;max-width: 280px; overflow-x: auto; overflow-y:  auto; text-align: left;'>{setting_content}</div>"""


def format_as_html(settings: dict):
    """
    Generates HTML content displaying the settings.
    Args:
        settings (dict): The dictionary containing the settings.
    Returns:
        str: The HTML content representing the settings.
    """
    return f"""
    <h3>Model Settings</h3>
    <p>sample_direc: {settings.get("sample_direc", "unknown")}</p>
    <p>model_type: {settings.get("model_type", "unknown")}</p>
    <p>implementation: {settings.get("implementation", "unknown")}</p>
    <p>otsu enabled: {settings.get("otsu", "unknown")}</p>
    <p>tta enabled: {settings.get("tta", "unknown")}</p>
    <h3>Extract Shorelines Settings</h3>
    <p>cloud_thresh: {settings.get("cloud_thresh", "unknown")}</p>
    <p>dist_clouds: {settings.get("dist_clouds", "unknown")}</p>
    <p>output_epsg: {settings.get("output_epsg", "unknown")}</p>
    <p>save_figure: {settings.get("save_figure", "unknown")}</p>
    <p>min_beach_area: {settings.get("min_beach_area", "unknown")}</p>
    <p>min_length_sl: {settings.get("min_length_sl", "unknown")}</p>
    <p>cloud_mask_issue: {settings.get("cloud_mask_issue", "unknown")}</p>
    <p>sand_color: {settings.get("sand_color", "unknown")}</p>
    <p>max_dist_ref: {settings.get("max_dist_ref", "unknown")}</p>
    <p>Drop intersection points not on transects (drop_intersection_pts): {settings.get("drop_intersection_pts", "unknown")}</p>
    <h3>Advanced Settings</h3>
    <p>along_dist: {settings.get("along_dist", "unknown")}</p>
    <p>min_points: {settings.get("min_points", "unknown")}</p>
    <p>max_std: {settings.get("max_std", "unknown")}</p>
    <p>max_range: {settings.get("max_range", "unknown")}</p>
    <p>min_chainage: {settings.get("min_chainage", "unknown")}</p>
    <p>multiple_inter: {settings.get("multiple_inter", "unknown")}</p>
    <p>prc_multiple: {settings.get("prc_multiple", "unknown")}</p>
    """
