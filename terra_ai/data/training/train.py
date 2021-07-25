"""
## Структура данных обучения
"""

import json
from typing import Any
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.errors import EnumMemberError

from ..mixins import BaseMixinData
from . import optimizers
from . import architectures
from .extra import OptimizerChoice, ArchitectureChoice


class OptimizerData(BaseMixinData):
    type: OptimizerChoice
    parameters: Any

    @property
    def parameters_dict(self) -> dict:
        __data = json.loads(self.parameters.main.json())
        __data.update(json.loads(self.parameters.extra.json()))
        return __data

    @validator("type", pre=True)
    def _validate_type(cls, value: OptimizerChoice) -> OptimizerChoice:
        if value not in list(OptimizerChoice):
            raise EnumMemberError(enum_values=list(OptimizerChoice))
        name = (
            value if isinstance(value, OptimizerChoice) else OptimizerChoice(value)
        ).name
        type_ = getattr(optimizers, getattr(optimizers.Optimizer, name))
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class ArchitectureData(BaseMixinData):
    type: ArchitectureChoice
    parameters: Any

    @validator("type", pre=True)
    def _validate_type(cls, value: ArchitectureChoice) -> ArchitectureChoice:
        if value not in list(ArchitectureChoice):
            raise EnumMemberError(enum_values=list(ArchitectureChoice))
        name = (
            value
            if isinstance(value, ArchitectureChoice)
            else ArchitectureChoice(value)
        ).name
        type_ = getattr(architectures, getattr(architectures.Architecture, name))
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class TrainData(BaseMixinData):
    batch: PositiveInt = 32
    epochs: PositiveInt = 20
    optimizer: OptimizerData = OptimizerData(type=OptimizerChoice.Adam)
    architecture: ArchitectureData = ArchitectureData(type=ArchitectureChoice.Basic)
