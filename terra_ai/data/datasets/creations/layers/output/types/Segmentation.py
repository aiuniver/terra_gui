from typing import Optional, List, Tuple, Union

from pydantic.types import DirectoryPath, PositiveInt, FilePath
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
    sources_paths: List[Union[DirectoryPath, FilePath]]
    cols_names: Optional[List[str]]
    classes_names: List[str]
    classes_colors: List[Color]
    mask_range: PositiveInt
    num_classes: Optional[PositiveInt]
    shape: Optional[Tuple[PositiveInt, ...]]
