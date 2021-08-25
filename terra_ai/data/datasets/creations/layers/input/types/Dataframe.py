from typing import Optional, List, Dict
from pydantic.types import PositiveInt

from ...extra import MinMaxScalerData
from .....extra import LayerScalerDataframeChoice
from ......types import confilepath


class ParametersData(MinMaxScalerData):
    sources_paths: List[confilepath(ext="csv")]

    separator: Optional[str]
    encoding: str = "utf-8"
    cols_names: Optional[List[str]]
    transpose: bool

    pad_sequences: Optional[bool]
    example_length: Optional[PositiveInt]
    xlen_step: Optional[bool]
    xlen: Optional[PositiveInt]
    step_len: Optional[PositiveInt]
    scaler: LayerScalerDataframeChoice = LayerScalerDataframeChoice.no_scaler

    StandardScaler: Optional[str]
    MinMaxScaler: Optional[str]
    Categorical: Optional[str]
    Categorical_ranges: Optional[str]
    cat_cols: Optional[Dict[str, str]]
    one_hot_encoding: Optional[str]
