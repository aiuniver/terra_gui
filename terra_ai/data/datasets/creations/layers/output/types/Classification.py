from typing import Optional, List, Union

from pydantic import PositiveInt, DirectoryPath, FilePath

from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
    cols_names: Optional[List[str]]
    one_hot_encoding: Optional[bool] = True
