from enum import Enum


class TaskTypeChoice(str, Enum):
    text_classification = "text_classification"
    regression = "regression"
    image_classification = "image_classification"
    object_detection = "object_detection"
    audio_classification = "audio_classification"
    text_segmentation = "text_segmentation"
    video_classification = "video_classification"
    video_segmentation = "video_segmentation"
    table_data_classification = "table_data_classification"
    time_series = "time_series"
    time_series_trend = "time_series_trend"
    time_series_classification = "time_series_classification"
    object_detection_tracking = "object_detection_tracking"
