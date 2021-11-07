from enum import Enum
from typing import Any

from ...mixins import BaseMixinData
from ..extra import BlockGroupChoice

from . import types


class BlockBaseData(BaseMixinData):
    pass


class BlockInputDataData(BlockBaseData):
    main: types.InputData.ParametersMainData = types.InputData.ParametersMainData()


class BlockOutputDataData(BlockBaseData):
    main: types.OutputData.ParametersMainData = types.OutputData.ParametersMainData()


class BlockModelData(BlockBaseData):
    main: types.Model.ParametersMainData = types.Model.ParametersMainData()


class BlockFunctionData(BlockBaseData):
    main: types.Function.ParametersMainData = types.Function.ParametersMainData()


class BlockCustomData(BlockBaseData):
    main: types.Custom.ParametersMainData = types.Custom.ParametersMainData()


Block = Enum(
    "Block",
    dict(map(lambda item: (item.name, f"Block{item.name}Data"), list(BlockGroupChoice))),
    type=str,
)
