from typing import Optional, List
from pydantic.types import PositiveInt

from ...extra import ParametersBaseData
from .....extra import LayerScalerChoice, LayerTaskTypeChoice


class ParametersData(ParametersBaseData):
    cols_names: Optional[List[str]]
    separator: Optional[str]
    length: PositiveInt
    depth: PositiveInt
    step: PositiveInt
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    task_type: LayerTaskTypeChoice = LayerTaskTypeChoice.timeseries
