"""
## Дополнительные структуры данных
"""

from enum import Enum
from typing import Optional, Tuple
from pydantic import validator, BaseModel


BYTES_UNITS = ["б", "Кб", "Мб", "Гб", "Тб", "Пб", "Эб", "Зб", "Иб"]


class FileSizeData(BaseModel):
    """
    Вес файла
    """

    value: int
    "Значение веса: `324133875`"
    short: Optional[float]
    "Короткое значение веса: `309.1181516647339`"
    unit: Optional[str]
    "Единицы измерения: `Мб`"

    def __init__(self, *args, **kwargs):
        kwargs = {
            "value": kwargs.get("value"),
            "short": 0,
            "unit": "",
        }
        super().__init__(*args, **kwargs)

    @staticmethod
    def __short_unit(value: int) -> Tuple[float, str]:
        divisor = 1024
        num = float(value)
        unit = BYTES_UNITS[0]
        for unit in BYTES_UNITS:
            if abs(num) < divisor:
                break
            num /= divisor
        return num, unit

    @validator("short", allow_reuse=True)
    def _validate_short(cls, _: float, **kwargs) -> float:
        short, unit = cls.__short_unit(kwargs.get("values", {}).get("value"))
        return short

    @validator("unit", allow_reuse=True)
    def _validate_unit(cls, _: str, **kwargs) -> str:
        short, unit = cls.__short_unit(kwargs.get("values", {}).get("value"))
        return unit


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
