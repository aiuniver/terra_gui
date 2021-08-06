from typing import Optional, List, Union
from pydantic.types import FilePath, PositiveInt, DirectoryPath
from pydantic import PositiveInt

from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
    separator: Optional[str]
    cols_names: Optional[List[str]]
    one_hot_encoding: Optional[bool] = True
    categorical: Optional[bool] = True
    categorical_ranges: Optional[bool] = True
    auto_ranges: Optional[bool] = True
    ranges: Optional[str]