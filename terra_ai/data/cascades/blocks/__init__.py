from enum import Enum
from typing import Any

from ...mixins import BaseMixinData
from ..extra import BlockGroupChoice

from . import types


class BlockBaseData(BaseMixinData):
    pass


class BlockInputDataData(BlockBaseData, types.InputData.ParametersData):
    pass


class BlockOutputDataData(BlockBaseData, types.OutputData.ParametersData):
    pass


class BlockModelData(BlockBaseData, types.Model.ParametersData):
    pass


class BlockFunctionData(BlockBaseData, types.Function.ParametersData):
    pass


class BlockCustomData(BlockBaseData, types.Custom.ParametersData):
    pass


Block = Enum(
    "Block",
    dict(map(lambda item: (item.name, f"Block{item.name}Data"), list(BlockGroupChoice))),
    type=str,
)
