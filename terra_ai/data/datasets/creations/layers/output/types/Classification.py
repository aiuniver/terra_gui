from typing import Optional, List
from pydantic import validator

from .....extra import LayerTypeProcessingClassificationChoice
from ......mixins import BaseMixinData
from ......types import confilepath


class ParametersData(BaseMixinData):
    sources_paths: List[confilepath(ext="csv")]
    one_hot_encoding: bool = True
    type_processing: LayerTypeProcessingClassificationChoice
    ranges: Optional[str]

    cols_names: Optional[List[str]]

    @validator("type_processing")
    def _validate_type_processing(
        cls, value: LayerTypeProcessingClassificationChoice
    ) -> LayerTypeProcessingClassificationChoice:
        if value == LayerTypeProcessingClassificationChoice.ranges:
            cls.__fields__["ranges"].required = True
        return value
