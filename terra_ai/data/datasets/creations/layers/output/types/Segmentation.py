from typing import Optional, List, Tuple

from pydantic.types import DirectoryPath, PositiveInt
from pydantic.color import Color

from ...extra import FileInfo
from ......mixins import BaseMixinData, UniqueListMixin


class MaskSegmentationData(BaseMixinData):
    name: str
    color: Color


class MasksSegmentationList(UniqueListMixin):
    class Meta:
        source = MaskSegmentationData
        identifier = "name"


class ParametersData(BaseMixinData):
    # image_path???
    file_info: FileInfo
    classes_names: List[str]
    classes_colors: List[Color]
    mask_range: PositiveInt
    num_classes: Optional[PositiveInt]
    shape: Optional[Tuple[PositiveInt, ...]]

