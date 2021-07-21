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


class LossChoice(str, Enum):
    CategoricalCrossentropy = "CategoricalCrossentropy"
    BinaryCrossentropy = "BinaryCrossentropy"
    MSE = "MSE"
    SquaredHinge = "SquaredHinge"
    Hinge = "Hinge"
    CategoricalHinge = "CategoricalHinge"
    SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    KullbackLeiblerDivergence = "KullbackLeiblerDivergence"
    Poisson = "Poisson"
    MAE = "MAE"
    Mape = "Mape"
    MSLE = "MSLE"
    LogCosh = "LogCosh"
    CosineSimilarity = "CosineSimilarity"


class MetricChoice(str, Enum):
    Accuracy = "Accuracy"
    BinaryAccuracy = "BinaryAccuracy"
    BinaryCrossentropy = "BinaryCrossentropy"
    CategoricalAccuracy = "CategoricalAccuracy"
    CategoricalCrossentropy = "CategoricalCrossentropy"
    SparseCategoricalAccuracy = "SparseCategoricalAccuracy"
    SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    TopKCategoricalAccuracy = "TopKCategoricalAccuracy"
    SparseTopKCategoricalAccuracy = "SparseTopKCategoricalAccuracy"
    Hinge = "Hinge"
    KullbackLeiblerDivergence = "KullbackLeiblerDivergence"
    Poisson = "Poisson"
    DiceCoef = "DiceCoef"
    MeanIOU = "MeanIOU"
    MAE = "MAE"
    MSE = "MSE"
    Mape = "Mape"
    MSLE = "MSLE"
    LogCosh = "LogCosh"
    CosineSimilarity = "CosineSimilarity"


class TaskGroupData(BaseMixinData):
    task: TaskChoice
    losses: List[LossChoice] = []
    metrics: List[MetricChoice] = []


class TasksGroupsList(UniqueListMixin):
    class Meta:
        source = TaskGroupData
        identifier = "task"
