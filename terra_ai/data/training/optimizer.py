"""
## Структура данных оптимайзера
"""

from typing import Optional, Any, Union
from pydantic import validator
from pydantic.errors import EnumMemberError

from ..mixins import BaseMixinData
from .extra import OptimizerTypeChoice
from . import parameters


class OptimizerData(BaseMixinData):
    """
    Параметры оптимайзера
    """

    type: OptimizerTypeChoice
    "Тип оптимайзера"
    parameters: Optional[Any]
    "Параметры оптимайзера"

    @validator("type", allow_reuse=True, pre=True)
    def _validate_type(cls, value: OptimizerTypeChoice) -> OptimizerTypeChoice:
        if not hasattr(OptimizerTypeChoice, value):
            raise EnumMemberError(enum_values=list(OptimizerTypeChoice))
        type_ = getattr(parameters, getattr(parameters.OptimizerParametersType, value))
        cls.__fields__["parameters"].type_ = type_
        cls.__fields__["parameters"].required = True
        return value

    @validator("parameters", allow_reuse=True)
    def _validate_parameters(
        cls, value: Any, **kwargs
    ) -> Union[parameters.OptimizerParametersTypeUnion]:
        return kwargs.get("field").type_(**value)
