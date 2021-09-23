from typing import List

from ...extra import MinMaxScalerData, ColumnProcessingData
from .....extra import LayerScalerRegressionChoice
from ......types import confilepath


class ParametersData(MinMaxScalerData, ColumnProcessingData):
    sources_paths: List[confilepath(ext="csv")]
    scaler: LayerScalerRegressionChoice
