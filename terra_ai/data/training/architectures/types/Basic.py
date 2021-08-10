from typing import Optional
from pydantic import validator

from ....mixins import BaseMixinData
from ....exceptions import ValueNotInListException
from ...outputs import OutputsList
from ...checkpoint import CheckpointData


class ParametersData(BaseMixinData):
    outputs: OutputsList = OutputsList()
    checkpoint: Optional[CheckpointData]

    @validator("checkpoint")
    def _validate_checkpoint(cls, value: CheckpointData, values) -> CheckpointData:
        if value is None:
            return value
        __available = values.get("outputs").ids
        if value.layer not in __available:
            raise ValueNotInListException(value.layer, __available)
        return value



