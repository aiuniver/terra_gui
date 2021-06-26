"""
## Дополнительные структуры данных
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class FileSizeData(BaseModel):
    """
    Вес файла
    """

    value: int
    "Значение веса: `324133875`"
    short: Optional[float]
    "Короткое значение веса: `309.12`"
    unit: Optional[str]
    "Единицы измерения: `Мб`"

    def __init__(self, *args, **kwargs):
        kwargs = {
            "value": kwargs.get("value"),
        }
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, *args, **kwargs):
        value = super().validate(*args)
        divisor = 1024
        units = ["б", "Кб", "Мб", "Гб", "Тб", "Пб", "Эб", "Зб", "Иб"]
        num = float(value.value)
        for unit in units:
            if abs(num) < divisor:
                value.short = num
                value.unit = unit
                break
            num /= divisor
        return value


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
