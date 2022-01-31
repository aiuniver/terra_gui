from enum import Enum

from ..extra import ArchitectureChoice
from . import types


class ArchitectureBasicData(types.Basic.ParametersData):
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


class ArchitectureYoloV3Data(types.YoloV3.ParametersData):
    pass


class ArchitectureYoloV4Data(types.YoloV4.ParametersData):
    pass


class ArchitectureTrackerData(ArchitectureBasicData):
    pass


class ArchitectureImageGANData(types.GAN.ParametersData):
    pass


class ArchitectureImageCGANData(types.GAN.ParametersData):
    pass


class ArchitectureTextToImageGANData(types.GAN.ParametersData):
    pass


class ArchitectureImageToImageGANData(types.GAN.ParametersData):
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
