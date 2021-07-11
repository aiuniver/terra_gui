from enum import Enum

from ...mixins import BaseMixinData
from ..extra import ArchitectureChoice
from . import types


class ArchitectureBaseData(BaseMixinData):
    pass


class ArchitectureBasicData(ArchitectureBaseData, types.Basic.ParametersData):
    pass


class ArchitectureYoloData(ArchitectureBaseData, types.Yolo.ParametersData):
    pass


Architecture = Enum(
    "Architecture",
    dict(
        map(
            lambda item: (item.name, f"Architecture{item.name}Data"),
            list(ArchitectureChoice),
        )
    ),
    type=str,
)
