import os
import re
import json

from pathlib import Path
from typing import Optional
from pydantic import validator
from transliterate import slugify

from terra_ai.exceptions.cascades import CascadeAlreadyExistsException
from terra_ai.data.cascades.block import BlocksList
from ... import settings
from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..types import confilepath, Base64Type


class CascadeLoadData(BaseMixinData):
    value: confilepath(ext=settings.CASCADE_EXT)


class CascadeDetailsData(AliasMixinData):
    name: Optional[str]
    image: Optional[Base64Type]
    blocks: BlocksList = []

    def save(self, path: Path, name: str, image: Base64Type, overwrite: bool = False):
        path = Path(path, f"{name}.{settings.CASCADE_EXT}")
        if path.is_file():
            if overwrite:
                os.remove(path)
            else:
                raise CascadeAlreadyExistsException(name)
        self.name = name
        self.image = image
        self.alias = re.sub(r"([\-]+)", "_", slugify(name, language_code="ru"))
        with open(path, "w") as path_ref:
            json.dump(self.native(), path_ref)


class CascadeListData(BaseMixinData):
    value: confilepath(ext=settings.CASCADE_EXT)
    label: Optional[str]

    @validator("label", allow_reuse=True, always=True)
    def _validate_label(cls, value: str, values) -> str:
        file_path = Path(values.get("value"))
        if not file_path:
            return value
        return file_path.name.split(f".{settings.CASCADE_EXT}")[0]


class CascadesList(UniqueListMixin):
    class Meta:
        source = CascadeListData
        identifier = "label"

    def __init__(self, path: Path, data=None):
        path = Path(path)

        if not path.is_dir():
            super().__init__(data)
            return

        data = []

        for file_name in os.listdir(path):
            file_path = Path(path, file_name)

            if not file_path.is_file():
                continue

            if file_path.suffix != f".{settings.CASCADE_EXT}":
                continue

            file_name_split = file_path.name.split(file_path.suffix)
            data.append({"label": "".join(file_name_split[:-1]), "value": file_path})

        super().__init__(data)
