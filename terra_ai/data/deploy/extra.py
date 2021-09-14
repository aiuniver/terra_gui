from enum import Enum


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
    regression = "regression"
