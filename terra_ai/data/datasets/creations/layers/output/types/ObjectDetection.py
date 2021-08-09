from typing import Optional, List, Union
from pydantic.types import PositiveInt, DirectoryPath, FilePath

from ...image_augmentation import AugmentationData
from ......mixins import BaseMixinData
from .....extra import LayerNetChoice, LayerScalerChoice


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]