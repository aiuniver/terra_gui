from enum import Enum


class FieldTypeChoice(str, Enum):
    text = "text"
    text_array = "text_array"
    number = "number"
    checkbox = "checkbox"
    select = "select"
    select_group = "select_group"
    auto_complete = "auto_complete"
    select_creation_tasks = "select_creation_tasks"
    multiselect = "multiselect"
    multiselect_sources_paths = "multiselect_sources_paths"
    radio = "radio"
    button = "button"
    segmentation_manual = "segmentation_manual"
    segmentation_search = "segmentation_search"
    segmentation_annotation = "segmentation_annotation"
