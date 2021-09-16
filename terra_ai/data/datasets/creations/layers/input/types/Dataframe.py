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

    StandardScaler_cols: Optional[List[int]]
    MinMaxScaler_cols: Optional[List[int]]
    Categorical_cols: Optional[List[int]]
    Categorical_ranges_cols: Optional[List[int]]
    cat_cols: Optional[Dict[str, str]]
    one_hot_encoding_cols: Optional[List[int]]

    step: Optional[int] = None
    length: Optional[int] = None
    trend: Optional[bool] = False
    depth: Optional[int] = None
    y_cols: Optional[List[int]] = None