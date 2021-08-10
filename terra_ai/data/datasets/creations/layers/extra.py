from typing import List, Union
from pydantic.types import DirectoryPath, FilePath

from ....mixins import BaseMixinData


class ParametersBaseData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
