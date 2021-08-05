"""
## Дополнительные структуры данных
"""

import os

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
from pydantic import validator
from pydantic.types import FilePath, DirectoryPath
from pydantic.color import Color

from .mixins import BaseMixinData, UniqueListMixin
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


class FileManagerTypeChoice(str, Enum):
    folder = "folder"
    image = "image"
    audio = "audio"
    video = "video"
    table = "table"


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


class FileManagerItem(BaseMixinData):
    path: Union[FilePath, DirectoryPath]
    title: Optional[str]
    type: Optional[FileManagerTypeChoice]
    children: list = []

    @validator("title", always=True)
    def _validate_title(cls, value: str, values) -> str:
        fullpath = values.get("path")
        if not fullpath:
            return value
        return fullpath.name

    @validator("type", always=True)
    def _validate_type(cls, value: str, values) -> str:
        fullpath = values.get("path")
        if not fullpath:
            return value
        if os.path.isdir(fullpath):
            return FileManagerTypeChoice.folder
        else:
            return FileManagerTypeChoice.table

    @validator("children", always=True)
    def _validate_children(cls, value: list, values) -> list:
        fullpath = values.get("path")
        __items = []
        if fullpath and os.path.isdir(fullpath):
            for item in os.listdir(fullpath):
                __items.append(
                    FileManagerItem(**{"path": Path(fullpath, item).absolute()})
                )
        return __items

    def dict(self, **kwargs):
        __exclude = ["path"]
        if self.type != FileManagerTypeChoice.folder:
            __exclude.append("children")
        kwargs.update({"exclude": set(__exclude)})
        return super().dict(**kwargs)


class FileManagerList(UniqueListMixin):
    class Meta:
        source = FileManagerItem
        identifier = "title"
