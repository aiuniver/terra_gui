from typing import Optional
from pydantic.types import DirectoryPath, PositiveInt
from pydantic.color import Color

from ......mixins import BaseMixinData, UniqueListMixin


class MaskSegmentationData(BaseMixinData):
    name: str
    color: Color


class MasksSegmentationList(UniqueListMixin):
    class Meta:
        source = MaskSegmentationData
        identifier = "name"


class ParametersData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    mask_range: PositiveInt
    mask_assignment: MasksSegmentationList
