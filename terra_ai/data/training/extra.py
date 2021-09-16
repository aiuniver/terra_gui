"""
## Дополнительные структуры данных для оптимайзеров
"""

from enum import Enum
from typing import List

from ..mixins import BaseMixinData, UniqueListMixin


class ArchitectureChoice(str, Enum):
    Basic = "Basic"
    Yolo = "Yolo"


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


class TaskChoice(str, Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
    ObjectDetection = "ObjectDetection"


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


class MetricChoice(str, Enum):
    AUC = "AUC"
    Accuracy = "Accuracy"
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
    RootMeanSquaredError = "RootMeanSquaredError"
    SquaredHinge = "SquaredHinge"
    TopKCategoricalAccuracy = "TopKCategoricalAccuracy"
    TrueNegatives = "TrueNegatives"
    TruePositives = "TruePositives"
    SparseCategoricalAccuracy = "SparseCategoricalAccuracy"
    SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    SparseTopKCategoricalAccuracy = "SparseTopKCategoricalAccuracy"

    DiceCoef = "DiceCoef"


class TaskGroupData(BaseMixinData):
    task: TaskChoice
    losses: List[LossChoice] = []
    metrics: List[MetricChoice] = []


class TasksGroupsList(UniqueListMixin):
    class Meta:
        source = TaskGroupData
        identifier = "task"
