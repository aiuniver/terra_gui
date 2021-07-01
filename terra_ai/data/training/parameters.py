"""
## Структура данных для параметров оптимайзера
"""

import sys

from enum import Enum
from pydantic.types import PositiveFloat

from ..mixins import BaseMixinData
from .extra import OptimizerChoice
from . import optimizers_parameters as ops


class OptimizerParametersTypeDefaultData(BaseMixinData):
    pass


class OptimizerParametersTypeMainDefaultData(OptimizerParametersTypeDefaultData):
    name: OptimizerChoice
    learning_rate: PositiveFloat = 0.001


class OptimizerParametersTypeExtraDefaultData(OptimizerParametersTypeDefaultData):
    pass


class OptimizerParametersTypeMixinData(BaseMixinData):
    extra: OptimizerParametersTypeExtraDefaultData = (
        OptimizerParametersTypeExtraDefaultData()
    )


class OptimizerParametersTypeSGDData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.SGD)
    )
    extra: ops.SGD.ParametersExtraData = ops.SGD.ParametersExtraData()


class OptimizerParametersTypeRMSpropData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.RMSprop)
    )
    extra: ops.RMSprop.ParametersExtraData = ops.RMSprop.ParametersExtraData()


class OptimizerParametersTypeAdamData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.Adam)
    )
    extra: ops.Adam.ParametersExtraData = ops.Adam.ParametersExtraData()


class OptimizerParametersTypeAdadeltaData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.Adadelta)
    )
    extra: ops.Adadelta.ParametersExtraData = ops.Adadelta.ParametersExtraData()


class OptimizerParametersTypeAdagradData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.Adagrad)
    )
    extra: ops.Adagrad.ParametersExtraData = ops.Adagrad.ParametersExtraData()


class OptimizerParametersTypeAdamaxData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.Adamax)
    )
    extra: ops.Adamax.ParametersExtraData = ops.Adamax.ParametersExtraData()


class OptimizerParametersTypeNadamData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.Nadam)
    )
    extra: ops.Nadam.ParametersExtraData = ops.Nadam.ParametersExtraData()


class OptimizerParametersTypeFtrlData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name=OptimizerChoice.Ftrl)
    )
    extra: ops.Ftrl.ParametersExtraData = ops.Ftrl.ParametersExtraData()


OptimizerParametersType = Enum(
    "OptimizerParametersType",
    dict(
        map(
            lambda item: (item, f"OptimizerParametersType{item}Data"),
            list(OptimizerChoice),
        )
    ),
    type=str,
)
"""
Список возможных типов параметров оптимайзера
"""


OptimizerParametersTypeUnion = tuple(
    map(
        lambda item: getattr(sys.modules[__name__], item),
        OptimizerParametersType,
    )
)
"""
Список возможных типов данных для оптимайзера в виде классов
"""
