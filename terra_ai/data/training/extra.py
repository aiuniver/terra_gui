from enum import Enum
from typing import List, Tuple

from terra_ai.data.mixins import BaseMixinData, UniqueListMixin
from terra_ai.data.presets.training import TasksGroups


class StateStatusChoice(str, Enum):
    no_train = "no_train"  # Сетка в начальном состоянии (без обучений)
    training = "training"  # Сетка в обучении
    trained = "trained"  # Сетка обучена
    stopped = "stopped"  # Обучение Остановлена
    addtrain = "addtrain"  # Дообучение или возобновление после остановки
    kill = "kill"  # Завершение без действий и удаление


class LossGraphShowChoice(str, Enum):
    model = "model"
    classes = "classes"


class MetricGraphShowChoice(str, Enum):
    model = "model"
    classes = "classes"


class ExampleChoiceTypeChoice(str, Enum):
    best = "best"
    worst = "worst"
    seed = "seed"
    random = "random"


class DataTypeChoice(str, Enum):
    train = "train"
    val = "val"


class BalanceSortedChoice(str, Enum):
    descending = "descending"
    ascending = "ascending"
    alphabetic = "alphabetic"


class ArchitectureChoice(str, Enum):
    Basic = "Basic"
    ImageClassification = "ImageClassification"
    ImageSegmentation = "ImageSegmentation"
    ImageAutoencoder = "ImageAutoencoder"
    TextClassification = "TextClassification"
    TextSegmentation = "TextSegmentation"
    TextTransformer = "TextTransformer"
    DataframeClassification = "DataframeClassification"
    DataframeRegression = "DataframeRegression"
    Timeseries = "Timeseries"
    TimeseriesTrend = "TimeseriesTrend"
    AudioClassification = "AudioClassification"
    VideoClassification = "VideoClassification"
    VideoTracker = "VideoTracker"
    YoloV3 = "YoloV3"
    YoloV4 = "YoloV4"
    Speech2Text = "Speech2Text"
    Text2Speech = "Text2Speech"
    ImageGAN = "ImageGAN"
    ImageCGAN = "ImageCGAN"
    TextToImageGAN = "TextToImageGAN"
    ImageToImageGAN = "ImageToImageGAN"
    ImageSRGAN = "ImageSRGAN"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ArchitectureChoice))


class CheckpointIndicatorChoice(str, Enum):
    Val = "Val"
    Train = "Train"


# class CheckpointModeChoice(str, Enum):
#     Min = "Min"
#     Max = "Max"


class CheckpointTypeChoice(str, Enum):
    Metrics = "Metrics"
    Loss = "Loss"


class OptimizerChoice(str, Enum):
    SGD = "SGD"
    RMSprop = "RMSprop"
    Adam = "Adam"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adamax = "Adamax"
    Nadam = "Nadam"
    Ftrl = "Ftrl"

    @staticmethod
    def names() -> list:
        return list(map(lambda item: item.name, OptimizerChoice))

    @staticmethod
    def options() -> List[Tuple[str, str]]:
        return list(map(lambda item: (item.name, item.value), OptimizerChoice))


class TaskChoice(str, Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
    ObjectDetection = "ObjectDetection"
    Dataframe = "Dataframe"
    TextSegmentation = "TextSegmentation"
    TimeseriesTrend = "TimeseriesTrend"
    Generator = "Generator"
    Discriminator = "Discriminator"


class LossChoice(str, Enum):
    BinaryCrossentropy = "BinaryCrossentropy"
    CategoricalCrossentropy = "CategoricalCrossentropy"
    CategoricalHinge = "CategoricalHinge"
    CosineSimilarity = "CosineSimilarity"
    Hinge = "Hinge"
    Huber = "Huber"
    KLDivergence = "KLDivergence"
    LogCosh = "LogCosh"
    MeanAbsoluteError = "MeanAbsoluteError"
    MeanAbsolutePercentageError = "MeanAbsolutePercentageError"
    MeanSquaredError = "MeanSquaredError"
    MeanSquaredLogarithmicError = "MeanSquaredLogarithmicError"
    Poisson = "Poisson"
    SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    SquaredHinge = "SquaredHinge"
    YoloLoss = "YoloLoss"


class MetricChoice(str, Enum):
    AUC = "AUC"
    Accuracy = "Accuracy"
    BalancedRecall = "BalancedRecall"
    BalancedPrecision = "BalancedPrecision"
    BalancedFScore = "BalancedFScore"
    BinaryAccuracy = "BinaryAccuracy"
    BinaryCrossentropy = "BinaryCrossentropy"
    CategoricalAccuracy = "CategoricalAccuracy"
    CategoricalCrossentropy = "CategoricalCrossentropy"
    CategoricalHinge = "CategoricalHinge"
    CosineSimilarity = "CosineSimilarity"
    FScore = "FScore"
    FalseNegatives = "FalseNegatives"
    FalsePositives = "FalsePositives"
    Hinge = "Hinge"
    KLDivergence = "KLDivergence"
    LogCoshError = "LogCoshError"
    MeanAbsoluteError = "MeanAbsoluteError"
    MeanAbsolutePercentageError = "MeanAbsolutePercentageError"
    MeanIoU = "MeanIoU"
    MeanSquaredError = "MeanSquaredError"
    MeanSquaredLogarithmicError = "MeanSquaredLogarithmicError"
    Poisson = "Poisson"
    Precision = "Precision"
    Recall = "Recall"
    # RecallPercent = "RecallPercent"
    RootMeanSquaredError = "RootMeanSquaredError"
    SquaredHinge = "SquaredHinge"
    TopKCategoricalAccuracy = "TopKCategoricalAccuracy"
    TrueNegatives = "TrueNegatives"
    TruePositives = "TruePositives"
    SparseCategoricalAccuracy = "SparseCategoricalAccuracy"
    SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    SparseTopKCategoricalAccuracy = "SparseTopKCategoricalAccuracy"
    UnscaledMAE = "UnscaledMAE"
    DiceCoef = "DiceCoef"
    BalancedDiceCoef = "BalancedDiceCoef"
    mAP50 = "mAP50"
    mAP95 = "mAP95"
    PercentMAE = "PercentMAE"


class TaskGroupData(BaseMixinData):
    task: TaskChoice
    losses: List[LossChoice] = []
    metrics: List[MetricChoice] = []


class TasksGroupsList(UniqueListMixin):
    class Meta:
        source = TaskGroupData
        identifier = "task"


TasksRelations = TasksGroupsList(TasksGroups)
