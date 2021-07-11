from enum import Enum

from ...mixins import BaseMixinData
from ..extra import ArchitectureChoice


class ArchitectureBaseData(BaseMixinData):
    pass


class ArchitectureBasicData(ArchitectureBaseData):
    pass


class ArchitectureYoloData(ArchitectureBaseData):
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
