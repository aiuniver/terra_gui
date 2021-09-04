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
    Classification = "Классификация"
    Segmentation = "Сегментация"
    TextSegmentation = "Сегментация текстов"
    Regression = "Регрессия"
    Timeseries = "Временные ряды"
    ObjectDetection = "Обнаружение объектов"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerOutputTypeChoice))


class LayerNetChoice(str, Enum):
    convolutional = "Сверточная"
    linear = "Линейная"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerNetChoice))


class LayerScalerChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerChoice))


class LayerScalerImageChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"
    terra_image_scaler = "TerraImageScaler"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerImageChoice))


class LayerScalerVideoChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerVideoChoice))


class LayerScalerAudioChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerAudioChoice))


class LayerScalerRegressionChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), LayerScalerRegressionChoice)
        )


class LayerScalerTimeseriesChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"

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


class LayerTextModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerTextModeChoice))


class LayerVideoFillModeChoice(str, Enum):
    black_frames = "Черными кадрами"
    average_value = "Средним значением"
    last_frames = "Последними кадрами"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerVideoFillModeChoice))


class LayerVideoFrameModeChoice(str, Enum):
    keep_proportions = "Сохранить пропорции"
    stretch = "Растянуть"

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


class TrainingArchitectureChoice(str, Enum):
    Basic = "Базовая"
    Yolo = "Yolo"


class TrainingOptimizerChoice(str, Enum):
    SGD = "SGD"
    RMSprop = "RMSprop"
    Adam = "Adam"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adamax = "Adamax"
    Nadam = "Nadam"
    Ftrl = "Ftrl"


class TrainingCheckpointIndicatorChoice(str, Enum):
    Val = "Val"
    Train = "Train"


class TrainingCheckpointTypeChoice(str, Enum):
    Metrics = "Metrics"
    Loss = "Loss"


class TrainingCheckpointModeChoice(str, Enum):
    Min = "min"
    Max = "max"
