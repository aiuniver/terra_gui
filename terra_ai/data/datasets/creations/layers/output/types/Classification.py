from typing import Optional

from pydantic import PositiveInt

from ...extra import FileInfo
from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    # index???
    file_info: FileInfo
    one_hot_encoding: Optional[bool] = True
    num_classes: Optional[PositiveInt]
