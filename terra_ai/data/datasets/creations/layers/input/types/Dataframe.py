from typing import Optional, List, Dict
from pydantic.types import PositiveInt

from ...extra import ParametersBaseData
from .....extra import LayerScalerChoice


class ParametersData(ParametersBaseData):
    separator: Optional[str]
    encoding: str = "utf-8"
    cols_names: Optional[List[str]]
    transpose: bool

    pad_sequences: Optional[bool]
    example_length: Optional[PositiveInt]
    xlen_step: Optional[bool]
    xlen: Optional[PositiveInt]
    step_len: Optional[PositiveInt]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler

    StandardScaler: Optional[str]
    MinMaxScaler: Optional[str]
    Categorical: Optional[str]
    Categorical_ranges: Optional[str]
    cat_cols: Optional[Dict[str, str]]
    one_hot_encoding: Optional[str]
