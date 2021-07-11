from typing import Optional
from pydantic.types import PositiveInt

from ......mixins import BaseMixinData
from .....extra import LayerScalerChoice, LayerTaskTypeChoice


class ParametersData(BaseMixinData):
    length: PositiveInt
    y_cols: Optional[PositiveInt]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    task_type: LayerTaskTypeChoice = LayerTaskTypeChoice.timeseries
