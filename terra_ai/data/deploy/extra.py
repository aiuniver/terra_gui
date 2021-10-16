from enum import Enum


class EnvVersionChoice(str, Enum):
    v1 = "v1"


class CollectionClasses(str, Enum):
    image_segmentation = "ImageSegmentation"
    image_classification = "ImageClassification"
    text_segmentation = "TextTextSegmentation"
    text_classification = "TextClassification"
    video_classification = "VideoClassification"
    audio_classification = "AudioClassification"
    table_data_classification = "TableDataClassification"
    table_data_regression = "TableDataRegression"


class TaskTypeChoice(str, Enum):
    image_segmentation = "image_segmentation"
    image_classification = "image_classification"
    text_segmentation = "text_segmentation"
    text_classification = "text_classification"
    video_segmentation = "video_segmentation"
    video_classification = "video_classification"
    object_detection = "object_detection"
    object_detection_tracking = "object_detection_tracking"
    time_series = "time_series"
    time_series_trend = "time_series_trend"
    time_series_classification = "time_series_classification"
    audio_classification = "audio_classification"
    table_data_classification = "table_data_classification"
    table_data_regression = "table_data_regression"
    regression = "regression"
