from enum import Enum
from typing import List, Union

from pydantic import validator

from ....exceptions import ValueTypeException
from ....mixins import BaseMixinData
from ....types import confilepath


class PathTypeChoice(str, Enum):
    path_folder = "path_folder"
    path_file = "path_file"


class FileInfo(BaseMixinData):
    path_type: PathTypeChoice
    path: List[Union[str, confilepath(ext="csv")]]

    @validator("path", allow_reuse=True)
    def _validate_path_type_path(
            cls, path: Union[str, confilepath(ext="csv")], values
    ) -> Union[str, confilepath(ext="csv")]:
        path_type = values.get("path_type")
        if path_type == PathTypeChoice.path_file:
            if not isinstance(path, confilepath(ext="csv")):
                raise ValueTypeException(path, confilepath(ext="csv"))
        if path_type == PathTypeChoice.path_folder:
            if not isinstance(path, str):
                raise ValueTypeException(path, str)
        return path
