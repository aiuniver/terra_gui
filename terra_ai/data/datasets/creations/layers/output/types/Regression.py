from typing import Optional, List

from ...extra import SourcesPathsData
from .....extra import LayerScalerRegressionChoice


class ParametersData(SourcesPathsData):
    cols_names: Optional[List[str]]
    scaler: LayerScalerRegressionChoice = LayerScalerRegressionChoice.no_scaler
