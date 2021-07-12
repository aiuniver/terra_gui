from typing import Optional
from pydantic.types import FilePath, PositiveInt

from ......mixins import BaseMixinData
from .....extra import LayerScalerChoice


class ParametersData(BaseMixinData):
    file_path: Optional[FilePath]
    separator: Optional[str]
    encoding: str = "utf-8"
    x_cols: Optional[PositiveInt]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
