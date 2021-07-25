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
