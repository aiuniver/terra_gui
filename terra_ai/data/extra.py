"""
## Дополнительные структуры данных
"""

from enum import Enum
from typing import Optional, Tuple
from pydantic import validator, BaseModel
from pydantic.color import Color

from .mixins import BaseMixinData
from .types import ConstrainedFloatValueGe0, ConstrainedIntValueGe0


BYTES_UNITS = ["б", "Кб", "Мб", "Гб", "Тб", "Пб", "Эб", "Зб", "Иб"]


class HardwareAcceleratorChoice(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"


class HardwareAcceleratorColorChoice(str, Enum):
    CPU = "FF0000"
    GPU = "2EA022"
    TPU = "B8C324"


class HardwareAcceleratorData(BaseMixinData):
    type: HardwareAcceleratorChoice
    color: Optional[Color]

    @validator("color", always=True)
    def _validate_color(cls, value: str, values) -> str:
        __type = values.get("type")
        if not __type:
            return value
        return HardwareAcceleratorColorChoice[__type.name]


class FileSizeData(BaseMixinData):
    """
    Вес файла
    """

    value: ConstrainedIntValueGe0
    "Значение веса: `324133875`"
    short: Optional[ConstrainedFloatValueGe0]
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
    def _validate_short(cls, value: float, **kwargs) -> float:
        if value is None:
            return value
        short, unit = cls.__short_unit(kwargs.get("values", {}).get("value"))
        return short

    @validator("unit", allow_reuse=True)
    def _validate_unit(cls, value: str, **kwargs) -> str:
        if value is None:
            return value
        short, unit = cls.__short_unit(kwargs.get("values", {}).get("value"))
        return unit
