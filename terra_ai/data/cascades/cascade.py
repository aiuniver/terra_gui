import os
from pathlib import Path
from typing import Optional
from pydantic import validator

from terra_ai.data.cascades.block import BlocksList
from ... import settings
from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..types import confilepath, Base64Type


class CascadeLoadData(BaseMixinData):
    value: confilepath(ext=settings.CASCADE_EXT)
    

class CascadeDetailsData(AliasMixinData):
    name: str
    image: Base64Type
    blocks: BlocksList = []


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

        for file in os.listdir(path):
            file_path = Path(path, file)
            
            if not file_path.is_file():
                continue
            
            if not file_path.suffix == f".{settings.CASCADE_EXT}":
                continue

            file_name_split = file_path.name.split(file_path.suffix)
            data.append({
                "label": "".join(file_name_split[:-1]),
                "value": file_path
            })

        super().__init__(data)
