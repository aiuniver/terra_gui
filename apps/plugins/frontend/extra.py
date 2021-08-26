from enum import Enum


class FieldTypeChoice(str, Enum):
    text = "text"
    number = "number"
    checkbox = "checkbox"
    select = "select"
    select_group = "select_group"
    multiselect = "multiselect"
    radio = "radio"
    button = "button"
    segmentation_manual = "segmentation_manual"
    segmentation_search = "segmentation_search"
    segmentation_annotation = "segmentation_annotation"
