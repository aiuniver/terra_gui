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
    segmentation_classes = "segmentation_classes"
