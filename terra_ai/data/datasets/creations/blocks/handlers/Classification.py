from typing import Optional
from pydantic import validator, PositiveInt

from terra_ai.data.datasets.extra import LayerTypeProcessingChoice
from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    one_hot_encoding: bool = True
    type_processing: LayerTypeProcessingChoice
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
    def _validate_type_processing(cls, value, values, field):
        required = value == LayerTypeProcessingChoice.ranges
        cls.__fields__["ranges"].required = required
        return value
