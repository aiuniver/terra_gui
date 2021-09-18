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


class ColumnProcessingImageSegmentationData(types.ParametersImageSegmentationData):
    pass


class ColumnProcessingClassificationData(types.ParametersClassificationData):
    pass


class ColumnProcessingRegressionData(types.ParametersRegressionData):
    pass


class ColumnProcessingTimeseriesData(types.ParametersTimeseriesData):
    pass


class ColumnProcessingScalerData(types.ParametersScalerData):
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
