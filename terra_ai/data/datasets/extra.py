"""
## Дополнительные структуры данных для датасетов
"""

from enum import Enum


class SourceModeChoice(str, Enum):
    """
    Метод загрузки исходных данных для создания датасета
    """

    GoogleDrive = "GoogleDrive"
    "Использовать путь к архиву в папке Google-диска"
    URL = "URL"
    "Использовать ссылку на архив"
    Terra = "Terra"
    "Terra"


class LayerPrepareMethodChoice(str, Enum):
    embedding = "embedding"
    bag_of_words = "bag_of_words"
    word_to_vec = "word_to_vec"


class LayerTaskTypeChoice(str, Enum):
    timeseries = "timeseries"
    regression = "regression"


class LayerNetChoice(str, Enum):
    convolutional = "convolutional"
    linear = "linear"


class LayerAudioParameterChoice(str, Enum):
    audio_signal = "audio_signal"
    chroma_stft = "chroma_stft"
    mfcc = "mfcc"
    rms = "rms"
    spectral_centroid = "spectral_centroid"
    spectral_bandwidth = "spectral_bandwidth"
    spectral_rolloff = "spectral_rolloff"
    zero_crossing_rate = "zero_crossing_rate"


class LayerAudioResampleChoice(str, Enum):
    kaiser_best = "kaiser_best"
    kaiser_fast = "kaiser_fast"
    scipy = "scipy"


class LayerTextModeChoice(str, Enum):
    completely = "completely"
    length_and_step = "length_and_step"


class LayerAudioModeChoice(str, Enum):
    completely = "completely"
    length_and_step = "length_and_step"


class LayerAudioFillModeChoice(str, Enum):
    last_millisecond = "last_millisecond"
    loop = "loop"


class LayerScalerDefaultChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"
    standard_scaler = "standard_scaler"


class LayerScalerDataframeChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"
    standard_scaler = "standard_scaler"


class LayerScalerTimeseriesChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"
    standard_scaler = "standard_scaler"


class LayerScalerRegressionChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"
    standard_scaler = "standard_scaler"


class LayerScalerImageChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"
    terra_image_scaler = "terra_image_scaler"


class LayerScalerAudioChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"
    standard_scaler = "standard_scaler"


class LayerScalerVideoChoice(str, Enum):
    no_scaler = "no_scaler"
    min_max_scaler = "min_max_scaler"


class LayerVideoFillModeChoice(str, Enum):
    last_frames = "last_frames"
    loop = "loop"
    average_value = "average_value"


class LayerVideoModeChoice(str, Enum):
    completely = "completely"
    length_and_step = "length_and_step"


class LayerVideoFrameModeChoice(str, Enum):
    keep_proportions = "keep_proportions"
    stretch = "stretch"
    # stretch = "stretch"
    fit = "fit"
    cut = "cut"


class LayerImageFrameModeChoice(str, Enum):
    stretch = "stretch"
    fit = "fit"
    cut = "cut"


class LayerTypeProcessingClassificationChoice(str, Enum):
    categorical = "categorical"
    ranges = "ranges"


class LayerObjectDetectionModelChoice(str, Enum):
    yolo = "yolo"
    ssd = "ssd"
    fast_r_cnn = "fast_r_cnn"
    mask_r_cnn = "mask_r_cnn"


class LayerYoloChoice(str, Enum):
    v3 = "v3"
    v4 = "v4"


class LayerODDatasetTypeChoice(str, Enum):
    Yolo_terra = "Yolo_terra"
    Voc = "Voc"
    Kitti = "Kitti"
    Coco = "Coco"
    Yolov1 = "Yolov1"
    Udacity = "Udacity"


class LayerEncodingChoice(str, Enum):
    none = "none"
    ohe = "ohe"
    multi = "multi"


class DatasetGroupChoice(str, Enum):
    keras = "keras"
    terra = "terra"
    custom = "custom"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, DatasetGroupChoice))


class ColumnProcessingTypeChoice(str, Enum):
    """
    Типы обработчиков для колонок таблиц
    """

    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Video = "Video"
    Scaler = "Scaler"
    Classification = "Classification"
    Regression = "Regression"
    Segmentation = "Segmentation"
    Timeseries = "Timeseries"


class LayerInputTypeChoice(str, Enum):
    """
    Типы данных для `input`-слоев
    """

    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"
    Video = "Video"
    Classification = "Classification"
    Scaler = "Scaler"
    Raw = "Raw"


class LayerOutputTypeChoice(str, Enum):
    """
    Типы данных для `output`-слоев
    """

    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"
    Classification = "Classification"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
    TimeseriesTrend = "TimeseriesTrend"
    ObjectDetection = "ObjectDetection"
    Raw = "Raw"
