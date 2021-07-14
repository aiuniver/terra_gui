"""
## Параметры оптимайзеров
"""

from enum import Enum
from pydantic.types import PositiveFloat

from ...mixins import BaseMixinData
from ..extra import OptimizerChoice

from . import types


class OptimizerMainDefaultData(BaseMixinData):
    learning_rate: PositiveFloat = 0.001


class OptimizerExtraDefaultData(BaseMixinData):
    pass


class OptimizerBaseData(BaseMixinData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: OptimizerExtraDefaultData = OptimizerExtraDefaultData()


class OptimizerSGDData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.SGD.ParametersExtraData = types.SGD.ParametersExtraData()


class OptimizerRMSpropData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.RMSprop.ParametersExtraData = types.RMSprop.ParametersExtraData()


class OptimizerAdamData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.Adam.ParametersExtraData = types.Adam.ParametersExtraData()


class OptimizerAdadeltaData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.Adadelta.ParametersExtraData = types.Adadelta.ParametersExtraData()


class OptimizerAdagradData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.Adagrad.ParametersExtraData = types.Adagrad.ParametersExtraData()


class OptimizerAdamaxData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.Adamax.ParametersExtraData = types.Adamax.ParametersExtraData()


class OptimizerNadamData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.Nadam.ParametersExtraData = types.Nadam.ParametersExtraData()


class OptimizerFtrlData(OptimizerBaseData):
    main: OptimizerMainDefaultData = OptimizerMainDefaultData()
    extra: types.Ftrl.ParametersExtraData = types.Ftrl.ParametersExtraData()


Optimizer = Enum(
    "Optimizer",
    dict(
        map(
            lambda item: (item.name, f"Optimizer{item.name}Data"), list(OptimizerChoice)
        )
    ),
    type=str,
)
