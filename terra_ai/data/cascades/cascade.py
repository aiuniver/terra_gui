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
from ..types import condirpath, Base64Type


class CascadeLoadData(BaseMixinData):
    value: condirpath(ext=settings.CASCADE_EXT)


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
    value: condirpath(ext=settings.CASCADE_EXT)
    label: Optional[str]

    @validator("label", allow_reuse=True, always=True)
    def _validate_label(cls, value: str, values) -> str:
        dir_path = Path(values.get("value"))
        if not dir_path:
            return value
        return dir_path.name.split(f".{settings.CASCADE_EXT}")[0]


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

        for dir_name in os.listdir(path):
            dir_path = Path(path, dir_name)

            if not dir_path.is_dir():
                continue

            if dir_path.suffix != f".{settings.CASCADE_EXT}":
                continue

            if not Path(dir_path, settings.CASCADE_CONFIG).is_file():
                continue

            dir_name_split = dir_path.name.split(dir_path.suffix)
            data.append({"label": "".join(dir_name_split[:-1]), "value": dir_path})

        super().__init__(data)
