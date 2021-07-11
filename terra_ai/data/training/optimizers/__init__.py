"""
## Параметры оптимайзеров
"""

from enum import Enum
from pydantic.types import PositiveFloat

from ...mixins import BaseMixinData
from ..extra import OptimizerChoice

from . import (
    SGD,
    RMSprop,
    Adam,
    Adadelta,
    Adagrad,
    Adamax,
    Nadam,
    Ftrl,
)


class OptimizerMainDefaultData(BaseMixinData):
    learning_rate: PositiveFloat = 0.001


class OptimizerExtraDefaultData(BaseMixinData):
    pass


class OptimizerBaseData(BaseMixinData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: OptimizerExtraDefaultData = OptimizerExtraDefaultData()


class OptimizerSGDData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: SGD.ParametersExtraData = SGD.ParametersExtraData()


class OptimizerRMSpropData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: RMSprop.ParametersExtraData = RMSprop.ParametersExtraData()


class OptimizerAdamData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: Adam.ParametersExtraData = Adam.ParametersExtraData()


class OptimizerAdadeltaData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: Adadelta.ParametersExtraData = Adadelta.ParametersExtraData()


class OptimizerAdagradData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: Adagrad.ParametersExtraData = Adagrad.ParametersExtraData()


class OptimizerAdamaxData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: Adamax.ParametersExtraData = Adamax.ParametersExtraData()


class OptimizerNadamData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: Nadam.ParametersExtraData = Nadam.ParametersExtraData()


class OptimizerFtrlData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: Ftrl.ParametersExtraData = Ftrl.ParametersExtraData()


Optimizer = Enum(
    "Optimizer",
    dict(
        map(
            lambda item: (item.name, f"Optimizer{item.name}Data"), list(OptimizerChoice)
        )
    ),
    type=str,
)
