from typing import Optional, Union, List
from pydantic.types import PositiveInt, DirectoryPath, FilePath

from .....extra import LayerScalerChoice
from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
    cols_names: Optional[List[str]]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
