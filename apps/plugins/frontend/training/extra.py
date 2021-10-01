from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel


class ShowImagesChoice(str, Enum):
    Best = "Лучшие"
    Worst = "Худшие"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ShowImagesChoice))

    @staticmethod
    def names() -> list:
        return list(map(lambda item: item.name, ShowImagesChoice))

    @staticmethod
    def options() -> List[Tuple[str, str]]:
        return list(map(lambda item: (item.name, item.value), ShowImagesChoice))


class GroupData(BaseModel):
    label: str
    collapsable: bool
    collapsed: bool


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
    # SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
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
    # SparseCategoricalAccuracy = "SparseCategoricalAccuracy"
    # SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    # SparseTopKCategoricalAccuracy = "SparseTopKCategoricalAccuracy"
    DiceCoef = "DiceCoef"


class TaskChoice(str, Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, TaskChoice))

    @staticmethod
    def names() -> list:
        return list(map(lambda item: item.name, TaskChoice))

    @staticmethod
    def options() -> List[Tuple[str, str]]:
        return list(map(lambda item: (item.name, item.value), TaskChoice))


class ArchitectureChoice(str, Enum):
    Basic = "Базовая"
    Yolo = "Yolo"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ArchitectureChoice))

    @staticmethod
    def names() -> list:
        return list(map(lambda item: item.name, ArchitectureChoice))

    @staticmethod
    def options() -> List[Tuple[str, str]]:
        return list(map(lambda item: (item.name, item.value), ArchitectureChoice))


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
    def values() -> list:
        return list(map(lambda item: item.value, OptimizerChoice))

    @staticmethod
    def names() -> list:
        return list(map(lambda item: item.name, OptimizerChoice))

    @staticmethod
    def options() -> List[Tuple[str, str]]:
        return list(map(lambda item: (item.name, item.value), OptimizerChoice))
