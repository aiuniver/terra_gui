from typing import Optional, List

from ...extra import MinMaxScalerData
from .....extra import LayerScalerRegressionChoice
from ......types import confilepath


class ParametersData(MinMaxScalerData):
    sources_paths: List[confilepath(ext="csv")]
    scaler: LayerScalerRegressionChoice

    cols_names: Optional[List[str]]
