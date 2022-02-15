from typing import Optional
from pydantic import validator, PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerTypeProcessingClassificationChoice


class OptionsData(BaseMixinData):
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
