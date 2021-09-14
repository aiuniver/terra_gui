from typing import Optional, List, Dict
from pydantic.types import PositiveInt

from ...extra import MinMaxScalerData
from .....extra import LayerScalerDataframeChoice
from ......types import confilepath


class ParametersData(MinMaxScalerData):
    sources_paths: List[confilepath(ext="csv")]

    separator: Optional[str]
    encoding: str = "utf-8"
    cols_names: Optional[List[int]]
    transpose: bool

    pad_sequences: Optional[bool]
    example_length: Optional[PositiveInt]
    xlen_step: Optional[bool]
    xlen: Optional[PositiveInt]
    step_len: Optional[PositiveInt]
    scaler: LayerScalerDataframeChoice = LayerScalerDataframeChoice.no_scaler

    StandardScaler: Optional[List[int]]
    MinMaxScaler: Optional[List[int]]
    Categorical: Optional[List[int]]
    Categorical_ranges: Optional[List[int]]
    cat_cols: Optional[Dict[str, str]]
    one_hot_encoding: Optional[List[int]]

    step: Optional[int]
    length: Optional[int]
    trend: Optional[bool]
    depth: Optional[int]
    y_cols: Optional[List[int]]