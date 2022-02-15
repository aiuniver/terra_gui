from terra_ai.data.datasets.extra import LayerTypeProcessingClassificationChoice
from terra_ai.data.mixins import BaseMixinData
from pathlib import Path
from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt


class ParametersData(BaseMixinData):
    one_hot_encoding: bool = True
    type_processing: Optional[
        LayerTypeProcessingClassificationChoice
    ] = LayerTypeProcessingClassificationChoice.categorical
    ranges: Optional[str]

    # Внутренние параметры
    xlen_step: Optional[bool]
    xlen: Optional[PositiveInt] = None
    step_len: Optional[PositiveInt] = None
    separator: Optional[str]
    put: Optional[PositiveInt]

    @validator("type_processing")
    def _validate_type_processing(
        cls, value: LayerTypeProcessingClassificationChoice
    ) -> LayerTypeProcessingClassificationChoice:
        if value == LayerTypeProcessingClassificationChoice.ranges:
            cls.__fields__["ranges"].required = True
        return value

    @validator("sources_paths")
    def _validate_sources_paths(cls, value: List[Path]) -> List[Path]:
        if not len(value):
            return value
        if value[0].is_file():
            cls.__fields__["type_processing"].required = True
        return value
