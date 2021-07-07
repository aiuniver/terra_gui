from enum import Enum


class WidgetTypeChoice(str, Enum):
    text = "text"
    select = "select"
    number = "number"
    checkbox = "checkbox"
    radio = "radio"


class LayerInputTypeChoice(str, Enum):
    Images = "Images"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"


class LayerOutputTypeChoice(str, Enum):
    Images = "Images"
    Text = "Text"
    Audio = "Audio"
    Classification = "Classification"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
