"""
## Дополнительные структуры данных
"""

import os
import cv2
import numpy
import pandas
import base64

from PIL import Image
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
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


class FileManagerTypeBaseChoice(str, Enum):
    folder = "folder"
    image = "image"
    audio = "audio"
    video = "video"
    table = "table"
    text = "text"
    unknown = "unknown"


class FileManagerTypeChoice(str, Enum):
    folder = FileManagerTypeBaseChoice.folder.value
    jpg = FileManagerTypeBaseChoice.image.value
    jpeg = FileManagerTypeBaseChoice.image.value
    png = FileManagerTypeBaseChoice.image.value
    avi = FileManagerTypeBaseChoice.video.value
    csv = FileManagerTypeBaseChoice.table.value
    undefined = FileManagerTypeBaseChoice.unknown.value


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
    path: Optional[Union[FilePath, DirectoryPath]]
    title: Optional[str]
    type: Optional[FileManagerTypeChoice]
    children: list = []
    dragndrop: bool = False
    extra: Dict[str, Any] = {}

    @property
    def csv2data(self) -> Optional[dict]:
        if self.type != FileManagerTypeChoice.csv:
            return None
        dataframe = pandas.read_csv(self.path, nrows=5)
        data: list = numpy.where(dataframe.isnull(), None, dataframe).tolist()
        data.insert(0, list(dataframe.columns))
        return data

    @property
    def cover(self) -> Optional[str]:
        if self.type != FileManagerTypeChoice.folder:
            return None
        _types = []
        for item in self.children:
            if item.type.value == FileManagerTypeBaseChoice.image:
                _types.append(item.type.name)
        if not len(_types):
            return None
        _images = []
        for item in os.listdir(self.path):
            try:
                _type = FileManagerTypeChoice[item.split(".")[-1].lower()]
                if _type.value == FileManagerTypeBaseChoice.image:
                    _images.append(item)
            except KeyError:
                pass
        _image = Path(self.path, _images[0])
        _image_ext = _image.name.split(".")[-1].lower()
        with open(_image, "rb") as _image_ref:
            _image_base64 = base64.b64encode(_image_ref.read())
            return f'data:image/{_image_ext};base64,{_image_base64.decode("utf-8")}'

    @staticmethod
    def is_usable(path: Path) -> bool:
        return os.path.isdir(path) or (
            os.path.isfile(path) and str(path).lower().endswith(".csv")
        )

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
            __type = getattr(FileManagerTypeChoice, fullpath.suffix.lower()[1:], None)
            if __type is None:
                __type = FileManagerTypeChoice.undefined
            return __type

    @validator("children", always=True)
    def _validate_children(cls, value: list, values) -> list:
        fullpath = values.get("path")
        _extra = {}
        __items = []
        if fullpath and os.path.isdir(fullpath):
            files_grouped = {}
            for item in os.listdir(fullpath):
                item_path = Path(fullpath, item).absolute()
                if FileManagerItem.is_usable(item_path):
                    __items.append(FileManagerItem(**{"path": item_path}))
                else:
                    _ext = str(item_path).split(".")[-1].lower()
                    _count = files_grouped.get(_ext, {}).get("count", 0)
                    _count += 1
                    if not len(_extra.keys()):
                        try:
                            _type = FileManagerTypeChoice[_ext]
                        except KeyError:
                            _type = FileManagerTypeChoice.undefined
                        if _type == FileManagerTypeBaseChoice.image:
                            _im = Image.open(str(item_path.absolute()))
                            _extra = {
                                "width": _im.size[0],
                                "height": _im.size[1],
                            }
                        elif _type == FileManagerTypeBaseChoice.video:
                            _vid = cv2.VideoCapture(str(item_path.absolute()))
                            _extra = {
                                "width": _vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                "height": _vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            }
                    files_grouped.update({_ext: {"count": _count, "extra": _extra}})
            for _ext, _group_data in files_grouped.items():
                try:
                    _type = FileManagerTypeChoice[_ext]
                except KeyError:
                    _type = FileManagerTypeChoice.undefined
                __items.append(
                    FileManagerItem(
                        **{
                            "title": f'[{_group_data.get("count")}] {_ext}',
                            "type": _type,
                            "extra": _group_data.get("extra"),
                        }
                    )
                )
        return __items

    @validator("extra", always=True)
    def _validate_extra(cls, value: dict, values) -> list:
        if not value:
            value = {}
        if values.get("type") == FileManagerTypeBaseChoice.folder:
            for children in values.get("children"):
                if len(children.extra.keys()):
                    value.update(children.extra)
        return value

    def dict(self, **kwargs):
        __exclude = []
        if self.type != FileManagerTypeChoice.folder:
            __exclude.append("children")
        kwargs.update({"exclude": set(__exclude)})
        data = super().dict(**kwargs)
        if self.type == FileManagerTypeChoice.csv:
            data.update({"data": self.csv2data})
        else:
            data.update({"data": None})
        if self.type in [FileManagerTypeChoice.folder, FileManagerTypeChoice.csv]:
            data.update({"dragndrop": True})
        data.update({"cover": self.cover})
        return data


class FileManagerList(UniqueListMixin):
    class Meta:
        source = FileManagerItem
        identifier = "title"
