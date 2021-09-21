from typing import List, Tuple
from pydantic.types import FilePath
from pydantic.color import Color

from terra_ai.data.mixins import BaseMixinData, UniqueListMixin
from terra_ai.data.types import ConstrainedFloatValueGe0Le100


class BaseCollection(BaseMixinData):
    pass


class ImageClassificationCollectionData(BaseCollection):
    source: FilePath
    data: List[Tuple[str, ConstrainedFloatValueGe0Le100]]


class ImageClassificationCollectionList(UniqueListMixin):
    class Meta:
        source = ImageClassificationCollectionData
        identifier = "source"


class ImageSegmentationCollectionData(BaseCollection):
    source: FilePath
    segment: FilePath
    data: List[Tuple[str, Color]]


class ImageSegmentationCollectionList(UniqueListMixin):
    class Meta:
        source = ImageSegmentationCollectionData
        identifier = "source"
