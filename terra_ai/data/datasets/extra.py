"""
## Дополнительные структуры данных для датасетов
"""

from enum import Enum


class SourceModeChoice(str, Enum):
    """
    Метод загрузки исходных данных для создания датасета
    """

    google_drive = "google_drive"
    "Использовать путь к архиву в папке Google-диска"
    url = "url"
    "Использовать ссылку на архив"


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


class LayerInputTypeChoice(str, Enum):
    """
    Типы данных для `input`-слоев
    """

    Images = "Images"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"


class LayerOutputTypeChoice(str, Enum):
    """
    Типы данных для `output`-слоев
    """

    Images = "Images"
    Text = "Text"
    Audio = "Audio"
    Classification = "Classification"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
