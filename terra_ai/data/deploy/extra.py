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
        return DeployTypeDemoChoice(self.value).name


class DeployTypeDemoChoice(str, Enum):
    image_segmentation = DeployTypeChoice.ImageSegmentation.value
    image_classification = DeployTypeChoice.ImageClassification.value
    text_segmentation = DeployTypeChoice.TextSegmentation.value
    text_classification = DeployTypeChoice.TextClassification.value
    audio_classification = DeployTypeChoice.AudioClassification.value
    video_classification = DeployTypeChoice.VideoClassification.value
    table_data_regression = DeployTypeChoice.DataframeRegression.value
    table_data_classification = DeployTypeChoice.DataframeClassification.value
    time_series = DeployTypeChoice.Timeseries.value
    time_series_trend = DeployTypeChoice.TimeseriesTrend.value
    yolo_v3 = DeployTypeChoice.YoloV3.value
    yolo_v4 = DeployTypeChoice.YoloV4.value
