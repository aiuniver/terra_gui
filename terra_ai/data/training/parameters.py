"""
## Структура данных для параметров оптимайзера
"""

import sys

from enum import Enum

from ..mixins import BaseMixinData
from ..types import ConstrainedFloatValueGt0
from .extra import OptimizerTypeChoice
from . import optimizers_parameters as ops


class OptimizerParametersTypeDefaultData(BaseMixinData):
    pass


class OptimizerParametersTypeMainDefaultData(OptimizerParametersTypeDefaultData):
    name: OptimizerTypeChoice
    learning_rate: ConstrainedFloatValueGt0 = 0.001


class OptimizerParametersTypeExtraDefaultData(OptimizerParametersTypeDefaultData):
    pass


class OptimizerParametersTypeMixinData(BaseMixinData):
    extra: OptimizerParametersTypeExtraDefaultData = (
        OptimizerParametersTypeExtraDefaultData()
    )


class OptimizerParametersTypeSGDData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="SGD")
    )
    extra: ops.SGD.ParametersExtraData = ops.SGD.ParametersExtraData()


class OptimizerParametersTypeRMSpropData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="RMSprop")
    )
    extra: ops.RMSprop.ParametersExtraData = ops.RMSprop.ParametersExtraData()


class OptimizerParametersTypeAdamData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="Adam")
    )
    extra: ops.Adam.ParametersExtraData = ops.Adam.ParametersExtraData()


class OptimizerParametersTypeAdadeltaData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="Adadelta")
    )
    extra: ops.Adadelta.ParametersExtraData = ops.Adadelta.ParametersExtraData()


class OptimizerParametersTypeAdagradData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="Adagrad")
    )
    extra: ops.Adagrad.ParametersExtraData = ops.Adagrad.ParametersExtraData()


class OptimizerParametersTypeAdamaxData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="Adamax")
    )
    extra: ops.Adamax.ParametersExtraData = ops.Adamax.ParametersExtraData()


class OptimizerParametersTypeNadamData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="Nadam")
    )
    extra: ops.Nadam.ParametersExtraData = ops.Nadam.ParametersExtraData()


class OptimizerParametersTypeFtrlData(OptimizerParametersTypeMixinData):
    main: OptimizerParametersTypeMainDefaultData = (
        OptimizerParametersTypeMainDefaultData(name="Ftrl")
    )
    extra: ops.Ftrl.ParametersExtraData = ops.Ftrl.ParametersExtraData()


OptimizerParametersType = Enum(
    "OptimizerParametersType",
    dict(
        map(
            lambda item: (item, f"OptimizerParametersType{item}Data"),
            list(OptimizerTypeChoice),
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
