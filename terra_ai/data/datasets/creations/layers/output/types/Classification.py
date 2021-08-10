from typing import Optional, List

from ...extra import ParametersBaseData


class ParametersData(ParametersBaseData):
    separator: Optional[str]
    cols_names: Optional[List[str]]
    one_hot_encoding: Optional[bool] = True
    categorical: Optional[bool] = True
    categorical_ranges: Optional[bool]
    auto_ranges: Optional[bool]
    ranges: Optional[str]
