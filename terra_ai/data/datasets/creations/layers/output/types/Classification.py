from typing import Optional

from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    one_hot_encoding: Optional[bool] = True
