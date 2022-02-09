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


class ColumnProcessingObjectDetectionData(types.ParametersObjectDetectionData):
    pass


class ColumnProcessingRegressionData(types.ParametersRegressionData):
    pass


class ColumnProcessingSegmentationData(types.ParametersSegmentationData):
    pass


class ColumnProcessingTextSegmentationData(types.ParametersTextSegmentationData):
    pass


class ColumnProcessingTimeseriesData(types.ParametersTimeseriesData):
    pass


class ColumnProcessingImageGANData(types.ParametersImageGANData):
    pass


class ColumnProcessingImageCGANData(types.ParametersImageCGANData):
    pass


class ColumnProcessingTextToImageGANData(types.ParametersTextToImageGANData):
    pass


class ColumnProcessingNoiseData(types.ParametersNoiseData):
    pass


class ColumnProcessingGeneratorData(types.ParametersGeneratorData):
    pass


class ColumnProcessingDiscriminatorData(types.ParametersDiscriminatorData):
    pass


class ColumnProcessingTransformerData(types.ParametersTransformerData):
    pass


class ColumnProcessingTrackerData(types.ParametersTrackerData):
    pass


class ColumnProcessingText2SpeechData(types.ParametersText2SpeechData):
    pass


class ColumnProcessingSpeech2TextData(types.ParametersSpeech2TextData):
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
