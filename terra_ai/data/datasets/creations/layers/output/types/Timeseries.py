from typing import Optional, List
from pydantic.types import PositiveInt

from ...extra import ParametersBaseData
from .....extra import LayerScalerChoice, LayerTaskTypeChoice


class ParametersData(ParametersBaseData):
    cols_names: Optional[List[str]]
    trend: bool
    trend_limit: Optional[str]
    length: PositiveInt
    depth:  Optional[PositiveInt]
    step: PositiveInt
    scaler: Optional[LayerScalerChoice] = LayerScalerChoice.no_scaler
    task_type: LayerTaskTypeChoice = LayerTaskTypeChoice.timeseries
