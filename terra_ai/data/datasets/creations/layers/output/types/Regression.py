from typing import Optional
from pydantic.types import PositiveInt

from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    y_col: Optional[PositiveInt]
