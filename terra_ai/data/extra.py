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


class LayerInputTypeChoice(str, Enum):
    """
    Типы данных для `input`-слоев
    """

    images = "images"
    text = "text"
    audio = "audio"
    dataframe = "dataframe"


class LayerOutputTypeChoice(str, Enum):
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
