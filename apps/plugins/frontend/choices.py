from enum import Enum


class LayerInputTypeChoice(str, Enum):
    Image = "Изображения"
    Text = "Текст"
    Audio = "Аудио"
    Video = "Видео"
    Dataframe = "Таблицы"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerInputTypeChoice))


class LayerOutputTypeChoice(str, Enum):
    Image = "Изображения"
    Text = "Текст"
    Audio = "Аудио"
    Dataframe = "Таблицы"
    Classification = "Классификация"
    Segmentation = "Сегментация изображений"
    TextSegmentation = "Сегментация текстов"
    ObjectDetection = "Обнаружение объектов"
    Tracker = "Трекер (каскад)"
    TrackerImages = "Изображения (каскад)"
    Text2Speech = "Генерация речи (каскад)"
    Speech2Text = "Распознавание речи (каскад)"
    ImageGAN = "ImageGAN"
    ImageCGAN = "ImageCGAN"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerOutputTypeChoice))


class LayerNetChoice(str, Enum):
    convolutional = "Сверточная"
    linear = "Линейная"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerNetChoice))


class LayerImageModeChoice(str, Enum):
    stretch = "Растянуть"
    fit = "Вписать"
    cut = "Обрезать"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerImageModeChoice))


class LayerScalerChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerChoice))


class LayerScalerDefaultChoice(str, Enum):
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"
    no_scaler = "Не применять"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerDefaultChoice))


class LayerScalerImageChoice(str, Enum):
    min_max_scaler = "MinMaxScaler"
    terra_image_scaler = "TerraImageScaler"
    no_scaler = "Не применять"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerImageChoice))


class LayerScalerVideoChoice(str, Enum):
    min_max_scaler = "MinMaxScaler"
    no_scaler = "Не применять"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerVideoChoice))


class LayerScalerAudioChoice(str, Enum):
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"
    no_scaler = "Не применять"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerAudioChoice))


class LayerScalerRegressionChoice(str, Enum):
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"
    no_scaler = "Не применять"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), LayerScalerRegressionChoice)
        )


class LayerScalerTimeseriesChoice(str, Enum):
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"
    no_scaler = "Не применять"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), LayerScalerTimeseriesChoice)
        )


class LayerAudioModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerAudioModeChoice))


class LayerAudioFillModeChoice(str, Enum):
    last_millisecond = "Последней миллисекундой"
    loop = "Зациклить"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerAudioFillModeChoice))


class LayerAudioParameterChoice(str, Enum):
    audio_signal = "Audio signal"
    chroma_stft = "Chroma STFT"
    mfcc = "MFCC"
    rms = "RMS"
    spectral_centroid = "Spectral centroid"
    spectral_bandwidth = "Spectral bandwidth"
    spectral_rolloff = "Spectral roll-off"
    zero_crossing_rate = "Zero-crossing rate"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), LayerAudioParameterChoice)
        )


class LayerAudioResampleChoice(str, Enum):
    kaiser_best = "Kaiser best"
    kaiser_fast = "Kaiser fast"
    scipy = "Scipy"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerAudioResampleChoice))


class LayerTextModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerTextModeChoice))


class LayerVideoFillModeChoice(str, Enum):
    last_frames = "Последним кадром"
    loop = "Зациклить"
    average_value = "Средним значением"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerVideoFillModeChoice))


class LayerVideoFrameModeChoice(str, Enum):
    stretch = "Растянуть"
    fit = "Вписать"
    cut = "Обрезать"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), LayerVideoFrameModeChoice)
        )


class LayerVideoModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerVideoModeChoice))


class LayerPrepareMethodChoice(str, Enum):
    no_preparation = "Не применять"
    embedding = "Embedding"
    bag_of_words = "Bag of words"
    word_to_vec = "Word2Vec"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerPrepareMethodChoice))


class LayerDataframeAlignBaseMethodChoice(str, Enum):
    pad_sequences = "Pad sequences"
    xlen_step = "XLen step"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(
                lambda item: (item.name, item.value),
                LayerDataframeAlignBaseMethodChoice,
            )
        )


class LayerDefineClassesChoice(str, Enum):
    handmade = "Ручной ввод"
    autosearch = "Автоматический поиск"
    annotation = "Файл аннотации"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerDefineClassesChoice))


class LayerYoloVersionChoice(str, Enum):
    v3 = "V3"
    v4 = "V4"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerYoloVersionChoice))


class LayerODDatasetTypeChoice(str, Enum):
    Yolo_terra = "Yolo_terra"
    Voc = "Voc"
    Kitti = "Kitti"
    Coco = "Coco"
    Yolov1 = "Yolov1"
    Udacity = "Udacity"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerODDatasetTypeChoice))


class LayerTypeProcessingClassificationChoice(str, Enum):
    categorical = "По категориям"
    ranges = "По диапазонам"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(
                lambda item: (item.name, item.value),
                LayerTypeProcessingClassificationChoice,
            )
        )


class ColumnProcessingInputTypeChoice(str, Enum):
    Image = "Изображения"
    Text = "Текст"
    Audio = "Аудио"
    Video = "Видео"
    Classification = "Уникальные"
    Scaler = "Скейлер"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), ColumnProcessingInputTypeChoice)
        )


class ColumnProcessingOutputTypeChoice(str, Enum):
    Classification = "Классификация"
    Regression = "Регрессия"
    Segmentation = "Сегментация"
    Timeseries = "Временные ряды"
    GAN = "GAN"
    Transformer = "Трансформер"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), ColumnProcessingOutputTypeChoice)
        )


class DeployTypePageChoice(str, Enum):
    model = "Обучение"
    cascade = "Каскад"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.name, DeployTypePageChoice))


class BlockFunctionGroupChoice(str, Enum):
    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Video = "Video"
    Array = "Array"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    ObjectDetection = "ObjectDetection"


class BlockFunctionTypeChoice(str, Enum):
    ChangeType = "Изменение типа данных"
    ChangeSize = "Изменение размера данных"
    MinMaxScale = "Нормализация (скейлер)"
    CropImage = "Обрезать изображение"
    MaskedImage = "Наложение маски по классу на изображение"
    PlotMaskSegmentation = "Наложение маски всех классов по цветам"
    PutTag = "Расстановка тегов по вероятностям из модели"
    PostprocessBoxes = "Постобработка Yolo"
    PlotBboxes = "Наложение BBox на изображение"
    FilterClasses = "Фильтрация классов Service YoloV5"
