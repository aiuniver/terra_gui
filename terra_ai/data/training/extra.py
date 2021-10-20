"""
## Дополнительные структуры данных для оптимайзеров
"""

from enum import Enum
from typing import List, Tuple

from ..mixins import BaseMixinData, UniqueListMixin


class StateStatusChoice(str, Enum):
    no_train = "no_train"
    training = "training"
    trained = "trained"
    stopped = "stopped"
    addtrain = "addtrain"


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


class BalanceSortedChoice(str, Enum):
    descending = "descending"
    ascending = "ascending"
    alphabetic = "alphabetic"


class ArchitectureChoice(str, Enum):
    Basic = "Basic"
    ImageClassification = "ImageClassification"
    ImageSegmentation = "ImageSegmentation"
    TextClassification = "TextClassification"
    TextSegmentation = "TextSegmentation"
    DataframeClassification = "DataframeClassification"
    DataframeRegression = "DataframeRegression"
    Timeseries = "Timeseries"
    TimeseriesTrend = "TimeseriesTrend"
    AudioClassification = "AudioClassification"
    VideoClassification = "VideoClassification"
    YoloV3 = "YoloV3"
    YoloV4 = "YoloV4"


class CheckpointIndicatorChoice(str, Enum):
    Val = "Val"
    Train = "Train"


class CheckpointModeChoice(str, Enum):
    Min = "Min"
    Max = "Max"


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
    BinaryAccuracy = "BinaryAccuracy"
    BinaryCrossentropy = "BinaryCrossentropy"
    CategoricalAccuracy = "CategoricalAccuracy"
    CategoricalCrossentropy = "CategoricalCrossentropy"
    CategoricalHinge = "CategoricalHinge"
    CosineSimilarity = "CosineSimilarity"
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
    RecallPercent = "RecallPercent"
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
    mAP50 = "mAP50"
    mAP95 = "mAP95"


class TaskGroupData(BaseMixinData):
    task: TaskChoice
    losses: List[LossChoice] = []
    metrics: List[MetricChoice] = []


class TasksGroupsList(UniqueListMixin):
    class Meta:
        source = TaskGroupData
        identifier = "task"
