import os

from pathlib import Path
from typing import Any, Optional, Union, List
from pydantic import validator
from pydantic.types import FilePath, DirectoryPath

from terra_ai.data.types import AliasType
from terra_ai.data.mixins import BaseMixinData, UniqueListMixin

from .extra import FieldTypeChoice, FileManagerTypeChoice


class ListOptionData(BaseMixinData):
    value: str
    label: str


class ListOptgroupData(BaseMixinData):
    label: str
    items: List[ListOptionData]


class Field(BaseMixinData):
    type: FieldTypeChoice
    name: AliasType
    label: str
    parse: str
    value: Any = ""
    list: Optional[List[Union[ListOptionData, ListOptgroupData]]]


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
