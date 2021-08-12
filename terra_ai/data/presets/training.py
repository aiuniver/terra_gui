"""
Предустановки обучения
"""

from enum import Enum


class Task(str, Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"


class Loss(str, Enum):
    CategoricalCrossentropy = "CategoricalCrossentropy"
    BinaryCrossentropy = "BinaryCrossentropy"
    MSE = "MSE"
    SquaredHinge = "SquaredHinge"
    Hinge = "Hinge"
    CategoricalHinge = "CategoricalHinge"
    SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    KLDivergence = "KLDivergence"
    Poisson = "Poisson"
    MAE = "MAE"
    Mape = "Mape"
    MSLE = "MSLE"
    LogCosh = "LogCosh"
    CosineSimilarity = "CosineSimilarity"


class Metric(str, Enum):
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


TasksGroups = [
    {
        "task": Task.Classification,
        "losses": [
            Loss.CategoricalCrossentropy,
            Loss.BinaryCrossentropy,
            Loss.MSE,
            Loss.SquaredHinge,
            Loss.Hinge,
            Loss.CategoricalHinge,
            Loss.SparseCategoricalCrossentropy,
            Loss.KLDivergence,
            Loss.Poisson,
        ],
        "metrics": [
            Metric.Accuracy,
            Metric.BinaryAccuracy,
            Metric.BinaryCrossentropy,
            Metric.CategoricalAccuracy,
            Metric.CategoricalCrossentropy,
            Metric.SparseCategoricalAccuracy,
            Metric.SparseCategoricalCrossentropy,
            Metric.TopKCategoricalAccuracy,
            Metric.SparseTopKCategoricalAccuracy,
            Metric.Hinge,
            Metric.KullbackLeiblerDivergence,
            Metric.Poisson,
        ],
    },
    {
        "task": Task.Segmentation,
        "losses": [
            Loss.CategoricalCrossentropy,
            Loss.BinaryCrossentropy,
            Loss.SquaredHinge,
            Loss.Hinge,
            Loss.CategoricalHinge,
            Loss.SparseCategoricalCrossentropy,
            Loss.KLDivergence,
            Loss.Poisson,
        ],
        "metrics": [
            Metric.DiceCoef,
            Metric.MeanIOU,
            Metric.Accuracy,
            Metric.BinaryAccuracy,
            Metric.BinaryCrossentropy,
            Metric.CategoricalAccuracy,
            Metric.CategoricalCrossentropy,
            Metric.SparseCategoricalAccuracy,
            Metric.SparseCategoricalCrossentropy,
            Metric.TopKCategoricalAccuracy,
            Metric.SparseTopKCategoricalAccuracy,
            Metric.Hinge,
            Metric.KullbackLeiblerDivergence,
            Metric.Poisson,
        ],
    },
    {
        "task": Task.Regression,
        "losses": [
            Loss.MSE,
            Loss.MAE,
            Loss.Mape,
            Loss.MSLE,
            Loss.LogCosh,
            Loss.CosineSimilarity,
        ],
        "metrics": [
            Metric.Accuracy,
            Metric.MAE,
            Metric.MSE,
            Metric.Mape,
            Metric.MSLE,
            Metric.LogCosh,
            Metric.CosineSimilarity,
        ],
    },
    {
        "task": Task.Timeseries,
        "losses": [
            Loss.MSE,
            Loss.MAE,
            Loss.Mape,
            Loss.MSLE,
            Loss.LogCosh,
            Loss.CosineSimilarity,
        ],
        "metrics": [
            Metric.Accuracy,
            Metric.MAE,
            Metric.MSE,
            Metric.Mape,
            Metric.MSLE,
            Metric.LogCosh,
            Metric.CosineSimilarity,
        ],
    },
]
