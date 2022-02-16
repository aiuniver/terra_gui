from typing import Optional
from pydantic import validator, PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerTypeProcessingChoice


class OptionsData(BaseMixinData):
    one_hot_encoding: bool = True
    type_processing: LayerTypeProcessingChoice = LayerTypeProcessingChoice.categorical
    ranges: Optional[str]

    # Внутренние параметры
    xlen_step: Optional[bool]
    xlen: Optional[PositiveInt] = None
    step_len: Optional[PositiveInt] = None
    separator: Optional[str]

    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    depth: Optional[PositiveInt]

    @validator("type_processing")
    def _validate_type_processing(
        cls, value: LayerTypeProcessingChoice
    ) -> LayerTypeProcessingChoice:
        if value == LayerTypeProcessingChoice.ranges:
            cls.__fields__["ranges"].required = True
        return value
