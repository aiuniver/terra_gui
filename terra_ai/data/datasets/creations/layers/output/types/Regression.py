from typing import List, Optional

from ...extra import MinMaxScalerData, ColumnProcessingData
from .....extra import LayerScalerRegressionChoice
from ......types import confilepath
from pydantic.types import PositiveInt
from typing import Optional


class ParametersData(MinMaxScalerData, ColumnProcessingData):
    sources_paths: List[confilepath(ext="csv")]
    scaler: LayerScalerRegressionChoice
    put: Optional[PositiveInt]
