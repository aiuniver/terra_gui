from enum import Enum


class EnvVersionChoice(str, Enum):
    v1 = "v1"


class DeployTypeChoice(str, Enum):
    ImageSegmentation = "ImageSegmentation"
    ImageClassification = "ImageClassification"
    TextSegmentation = "TextSegmentation"
    TextClassification = "TextClassification"
    AudioClassification = "AudioClassification"
    VideoClassification = "VideoClassification"
    DataframeRegression = "DataframeRegression"
    DataframeClassification = "DataframeClassification"
    Timeseries = "Timeseries"
    TimeseriesTrend = "TimeseriesTrend"
    YoloV3 = "YoloV3"
    YoloV4 = "YoloV4"

    @property
    def demo(self) -> str:
        return DeployTypeDemoChoice[self.value].value


class DeployTypeDemoChoice(str, Enum):
    ImageSegmentation = "image_segmentation"
    ImageClassification = "image_classification"
    TextSegmentation = "text_segmentation"
    TextClassification = "text_classification"
    AudioClassification = "audio_classification"
    VideoClassification = "video_classification"
    DataframeRegression = "table_data_regression"
    DataframeClassification = "table_data_classification"
    Timeseries = "time_series"
    TimeseriesTrend = "time_series_trend"
    YoloV3 = "object_detection"
    YoloV4 = "object_detection"


class DeployTypePageChoice(str, Enum):
    cascade = "cascade"
    model = "model"
