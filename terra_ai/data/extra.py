"""
## Дополнительные структуры данных
"""

from enum import Enum
from pydantic import BaseModel


class SizeData(BaseModel):
    """
    Вес файла
    """

    value: float
    "Значение веса `34.56`"
    unit: str
    "Единицы измерения: `Мб`"


class DatasetSourceModeChoice(str, Enum):
    """
    Метод загрузки исходных данных для создания датасета
    """

    google_drive = "google_drive"
    "Использовать путь к архиву в папке Google-диска"
    url = "url"
    "Использовать ссылку на архив"


class InputTypeChoice(str, Enum):
    """
    Типы данных для `input`-слоев
    """

    images = "images"
    text = "text"
    audio = "audio"
    dataframe = "dataframe"


class OutputTypeChoice(str, Enum):
    """
    Типы данных для `output`-слоев
    """

    images = "images"
    text = "text"
    audio = "audio"
    classification = "classification"
    segmentation = "segmentation"
    text_segmentation = "text_segmentation"
    regression = "regression"
    timeseries = "timeseries"


class LayerNetChoice(str, Enum):
    Convolutional = "Convolutional"
    Linear = "Linear"


class LayerScalerChoice(str, Enum):
    NoScaler = "NoScaler"
    MinMaxScaler = "MinMaxScaler"


class LayerPrepareMethodChoice(str, Enum):
    embedding = "embedding"
    bag_of_words = "bag_of_words"
    word_to_vec = "word_to_vec"


class LayerTaskTypeChoice(str, Enum):
    timeseries = "timeseries"
    regression = "regression"
