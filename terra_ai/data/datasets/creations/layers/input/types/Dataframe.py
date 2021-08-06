from typing import Optional, List, Dict, Union
from pydantic.types import FilePath, PositiveInt, DirectoryPath

from ......mixins import BaseMixinData
from .....extra import LayerScalerChoice


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
    separator: Optional[str]
    encoding: str = "utf-8"
    cols_names: Optional[List[str]]
    transpose: bool
    trend: bool
    trend_limit: Optional[str]
    ohe_trend: Optional[bool]
    pad_sequences: Optional[bool]
    example_lengh: Optional[PositiveInt]
    xlen_step: Optional[bool]
    xlen: Optional[PositiveInt]
    step_len: Optional[PositiveInt]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    StandardScaler: Optional[str]
    MinMaxScaler: Optional[str]
    Categorical: Optional[str]
    Categorical_ranges: Optional[str]
    cols: Optional[Dict[PositiveInt, str]]
    one_hot_encoding: Optional[str]
