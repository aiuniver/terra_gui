from enum import Enum


class LayerInputTypeChoice(str, Enum):
    Image = "Изображения"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"
    Video = "Video"


class LayerOutputTypeChoice(str, Enum):
    Image = "Изображения"
    Text = "Text"
    Audio = "Audio"
    Classification = "Classification"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
    ObjectDetection = "ObjectDetection"


class LayerNetChoice(str, Enum):
    convolutional = "Сверточная"
    linear = "Линейная"


class LayerScalerChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"
