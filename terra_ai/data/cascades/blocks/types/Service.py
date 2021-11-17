from typing import Optional
from pydantic import PositiveInt, validator

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.blocks.extra import BlockServiceGroupChoice, BlockServiceTypeChoice


class ParametersMainData(BaseMixinData):
    group: BlockServiceGroupChoice
    type: BlockServiceTypeChoice
    max_age: Optional[PositiveInt] = 4
    min_hits: Optional[PositiveInt] = 4

    @validator("type")
    def _validate_type(cls, value: BlockServiceTypeChoice) -> BlockServiceTypeChoice:
        if value == BlockServiceTypeChoice.Sort:
            cls.__fields__["max_age"].required = True
            cls.__fields__["min_hits"].required = True
        if value == BlockServiceTypeChoice.DeepSort:
            cls.__fields__["max_age"].required = True
            cls.__fields__["min_hits"].required = True
     
        return value