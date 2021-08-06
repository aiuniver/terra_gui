from typing import Optional, List, Union
from pydantic.types import PositiveInt, DirectoryPath, FilePath

from ......mixins import BaseMixinData
from .....extra import LayerScalerChoice, LayerTaskTypeChoice


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
    length: PositiveInt
    depth: PositiveInt
    step: PositiveInt
    y_cols: Optional[str]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    task_type: LayerTaskTypeChoice = LayerTaskTypeChoice.timeseries
