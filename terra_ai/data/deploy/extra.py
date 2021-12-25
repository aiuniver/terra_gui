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
    VideoObjectDetection = "VideoObjectDetection"
    YoloV3 = "YoloV3"
    YoloV4 = "YoloV4"
    YoloV5 = "YoloV5"
    GoogleTTS = "GoogleTTS"
    Wav2Vec = "Wav2Vec"
    TinkoffAPI = "TinkoffAPI"

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
    VideoObjectDetection = "video_object_detection"
    YoloV3 = "object_detection"
    YoloV4 = "object_detection"
    YoloV5 = "object_detection"
    GoogleTTS = "text_to_audio"
    Wav2Vec = "audio_to_text"
    TinkoffAPI = "audio_to_text"


class DeployTypePageChoice(str, Enum):
    cascade = "cascade"
    model = "model"
