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


class LayerGroupChoice(str, Enum):
    inputs = "inputs"
    outputs = "outputs"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, LayerGroupChoice))


class LayerSelectTypeChoice(str, Enum):
    table = "table"
    folder = "folder"


class LayerTypeChoice(str, Enum):
    data = "data"
    handler = "handler"
    input = "input"
    output = "output"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, LayerTypeChoice))


class LayerPrepareMethodChoice(str, Enum):
    no_preparation = "no_preparation"
    embedding = "embedding"
    bag_of_words = "bag_of_words"
    word_to_vec = "word_to_vec"


# class LayerTaskTypeChoice(str, Enum):
#     timeseries = "timeseries"
#     regression = "regression"


class DatasetTaskTypeChoice(str, Enum):
    ImageClassification = "ImageClassification"  # 1 вход, 1 выход
    ImageSegmentation = "ImageSegmentation"  # 1 вход, 1 выход
    # ImageObjectDetection = 'ImageObjectDetection'  # 1 вход, 1(3+3) выход(ов)
    ImageGAN = "ImageGAN"
    ImageCGAN = "ImageCGAN"
    TextClassification = "TextClassification"  # 1 вход, 1 выход
    TextSegmentation = "TextSegmentation"  # 1 вход, 1 выход
    VideoClassification = "VideoClassification"  # 1 вход, 1 выход
    VideoSegmentation = "VideoSegmentation"  # 1 вход, 1 выход
    VideoTracker = "VideoTracker"  # 1 вход, 1 выход
    AudioClassification = "AudioClassification"  # * вход, 1 выход
    AudioSegmentation = "AudioSegmentation"  # * вход, 1 выход
    DataframeClassification = "DataframeClassification"  # *вход(ов), * выход(ов)
    DataframeRegression = "DataframeRegression"  # *вход(ов), * выход(ов)
    Timeseries = "Timeseries"  # 1 вход, 1 выход
    TimeseriesTrend = "TimeseriesTrend"  # 1 вход, 1 выход
    YoloV3 = "YoloV3"  # 1 вход, 1(3+3) выход(ов)
    YoloV4 = "YoloV4"  # 1 вход, 1(3+3) выход(ов)
    Text2Speech = "Text2Speech"
    Speech2Text = "Speech2Text"


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
    stretch = "stretch"
    fit = "fit"
    cut = "cut"


class LayerImageFrameModeChoice(str, Enum):
    stretch = "stretch"
    fit = "fit"
    cut = "cut"


class LayerTypeProcessingChoice(str, Enum):
    categorical = "categorical"
    ranges = "ranges"


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
    ObjectDetection = "ObjectDetection"
    Regression = "Regression"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    Timeseries = "Timeseries"
    ImageGAN = "ImageGAN"
    ImageCGAN = "ImageCGAN"
    TextToImageGAN = "TextToImageGAN"
    Noise = "Noise"
    Discriminator = "Discriminator"
    Generator = "Generator"
    Tracker = "Tracker"
    Text2Speech = "Text2Speech"
    Speech2Text = "Speech2Text"
    Transformer = "Transformer"


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
    Noise = "Noise"


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
    YoloV3 = "YoloV3"
    YoloV4 = "YoloV4"
    Raw = "Raw"
    Tracker = "Tracker"
    Speech2Text = "Speech2Text"
    Text2Speech = "Text2Speech"
    ImageGAN = "ImageGAN"
    ImageCGAN = "ImageCGAN"
    TextToImageGAN = "TextToImageGAN"
    Discriminator = "Discriminator"
    Generator = "Generator"


class LayerHandlerChoice(str, Enum):
    Audio = "Audio"
    Classification = "Classification"
    Discriminator = "Discriminator"
    Generator = "Generator"
    Image = "Image"
    ImageCGAN = "ImageCGAN"
    ImageGAN = "ImageGAN"
    Noise = "Noise"
    YoloV3 = "YoloV3"
    YoloV4 = "YoloV4"
    Regression = "Regression"
    Scaler = "Scaler"
    Segmentation = "Segmentation"
    Speech2Text = "Speech2Text"
    Text = "Text"
    Text2Speech = "Text2Speech"
    TextSegmentation = "TextSegmentation"
    TextTransformer = "TextTransformer"
    TextToImageGAN = "TextToImageGAN"
    Timeseries = "Timeseries"
    TimeseriesTrend = "TimeseriesTrend"
    Tracker = "Tracker"
    Transformer = "Transformer"
    Video = "Video"

    @staticmethod
    def inputs() -> list:
        return [
            LayerHandlerChoice.Image,
            LayerHandlerChoice.Audio,
            LayerHandlerChoice.Classification,
            LayerHandlerChoice.Noise,
            LayerHandlerChoice.Scaler,
            LayerHandlerChoice.Text,
            LayerHandlerChoice.Video,
        ]

    @staticmethod
    def outputs() -> list:
        return [
            LayerHandlerChoice.Classification,
            LayerHandlerChoice.Discriminator,
            LayerHandlerChoice.Generator,
            LayerHandlerChoice.ImageCGAN,
            LayerHandlerChoice.ImageGAN,
            LayerHandlerChoice.YoloV3,
            LayerHandlerChoice.YoloV4,
            LayerHandlerChoice.Regression,
            LayerHandlerChoice.Segmentation,
            LayerHandlerChoice.Speech2Text,
            LayerHandlerChoice.Text2Speech,
            LayerHandlerChoice.TextSegmentation,
            LayerHandlerChoice.TextToImageGAN,
            LayerHandlerChoice.Timeseries,
            LayerHandlerChoice.TimeseriesTrend,
            LayerHandlerChoice.Tracker,
            LayerHandlerChoice.Transformer,
            LayerHandlerChoice.TextTransformer,
        ]
