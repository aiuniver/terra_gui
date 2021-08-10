from typing import Optional, List

from ...extra import ParametersBaseData
from .....extra import LayerScalerChoice


class ParametersData(ParametersBaseData):
    cols_names: Optional[List[str]]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
