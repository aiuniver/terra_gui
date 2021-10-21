from enum import Enum


class EnvVersionChoice(str, Enum):
    v1 = "v1"


class DeployTypeChoice(str, Enum):
    ImageSegmentation = "ImageSegmentation"
    ImageClassification = "ImageClassification"
    TextSegmentation = "TextSegmentation"
    TextClassification = "TextClassification"
    AudioClassification = "AudioClassification"
    DataframeRegression = "DataframeRegression"
    DataframeClassification = "DataframeClassification"
