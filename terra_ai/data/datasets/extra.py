"""
## Дополнительные структуры данных для датасетов
"""

from enum import Enum


class SourceModeChoice(str, Enum):
    """
    Метод загрузки исходных данных для создания датасета
    """

    GoogleDrive = "GoogleDrive"
    "Использовать путь к архиву в папке Google-диска"
    URL = "URL"
    "Использовать ссылку на архив"
    Terra = "Terra"
    "Terra"


class LayerPrepareMethodChoice(str, Enum):
    embedding = "embedding"
    bag_of_words = "bag_of_words"
    word_to_vec = "word_to_vec"


class LayerTaskTypeChoice(str, Enum):
    timeseries = "timeseries"
    regression = "regression"


class LayerNetChoice(str, Enum):
    convolutional = "convolutional"
    linear = "linear"


class LayerScalerChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"


class DatasetGroupChoice(str, Enum):
    keras = "keras"
    custom = "custom"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, DatasetGroupChoice))


class LayerInputTypeChoice(str, Enum):
    """
    Типы данных для `input`-слоев
    """

    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"
    Video = "Video"


class LayerOutputTypeChoice(str, Enum):
    """
    Типы данных для `output`-слоев
    """

    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Classification = "Classification"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
    ObjectDetection = "ObjectDetection"
