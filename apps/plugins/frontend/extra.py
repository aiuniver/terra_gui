from enum import Enum


class FieldTypeChoice(str, Enum):
    text = "text"
    number = "number"
    checkbox = "checkbox"
    select = "select"
    select_group = "select_group"
    multiselect = "multiselect"
    radio = "radio"


class FileManagerTypeChoice(str, Enum):
    folder = "folder"
    image = "image"
    audio = "audio"
    video = "video"
    table = "table"
