from enum import Enum

from terra_ai.data.datasets.extra import ColumnProcessingTypeChoice

from . import types


class ColumnProcessingImageData(types.ParametersImageData):
    pass


class ColumnProcessingTextData(types.ParametersTextData):
    pass


class ColumnProcessingAudioData(types.ParametersAudioData):
    pass


class ColumnProcessingVideoData(types.ParametersVideoData):
    pass


class ColumnProcessingScalerData(types.ParametersScalerData):
    pass


class ColumnProcessingClassificationData(types.ParametersClassificationData):
    pass


class ColumnProcessingRegressionData(types.ParametersRegressionData):
    pass


class ColumnProcessingSegmentationData(types.ParametersSegmentationData):
    pass


class ColumnProcessingTimeseriesData(types.ParametersTimeseriesData):
    pass


class ColumnProcessingGANData(types.ParametersGANData):
    pass


class ColumnProcessingCGANData(types.ParametersCGANData):
    pass


class ColumnProcessingTextToImageGANData(types.ParametersTextToImageGANData):
    pass


class ColumnProcessingNoiseData(types.ParametersNoiseData):
    pass


class ColumnProcessingGeneratorData(types.ParametersGeneratorData):
    pass


class ColumnProcessingDiscriminatorData(types.ParametersDiscriminatorData):
    pass


ColumnProcessing = Enum(
    "ColumnProcessing",
    dict(
        map(
            lambda item: (item.name, f"ColumnProcessing{item.name}Data"),
            list(ColumnProcessingTypeChoice),
        )
    ),
    type=str,
)
