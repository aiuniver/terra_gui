from enum import Enum

from ...mixins import BaseMixinData
from ..extra import ArchitectureChoice
from . import types


class ArchitectureBaseData(BaseMixinData):
    pass


class ArchitectureBasicData(ArchitectureBaseData, types.Basic.ParametersData):
    pass


class ArchitectureImageClassificationData(ArchitectureBasicData):
    pass


class ArchitectureImageSegmentationData(ArchitectureBasicData):
    pass


class ArchitectureTextClassificationData(ArchitectureBasicData):
    pass


class ArchitectureTextSegmentationData(ArchitectureBasicData):
    pass


class ArchitectureDataframeClassificationData(ArchitectureBasicData):
    pass


class ArchitectureDataframeRegressionData(ArchitectureBasicData):
    pass


class ArchitectureTimeseriesData(ArchitectureBasicData):
    pass


class ArchitectureTimeseriesTrendData(ArchitectureBasicData):
    pass


class ArchitectureAudioClassificationData(ArchitectureBasicData):
    pass


class ArchitectureVideoClassificationData(ArchitectureBasicData):
    pass


class ArchitectureYoloV3Data(ArchitectureBasicData, types.YoloV3.ParametersData):
    pass


class ArchitectureYoloV4Data(ArchitectureBasicData, types.YoloV4.ParametersData):
    pass


class ArchitectureTrackerData(ArchitectureBasicData):
    pass


class ArchitectureGANData(ArchitectureBasicData):
    pass


class ArchitectureCGANData(ArchitectureBasicData):
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
