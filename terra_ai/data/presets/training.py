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


class Metric(str, Enum):
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


TasksGroups = [
    {
        "task": Task.Classification,
        "losses": [
            Loss.BinaryCrossentropy,
            Loss.CategoricalCrossentropy,
            Loss.CategoricalHinge,
            # Loss.CosineSimilarity,
            Loss.Hinge,
            Loss.Huber,
            Loss.KLDivergence,
            Loss.LogCosh,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredError,
            Loss.MeanSquaredLogarithmicError,
            Loss.Poisson,
            Loss.SparseCategoricalCrossentropy,
            Loss.SquaredHinge
        ],
        "metrics": [
            Metric.AUC,
            Metric.Accuracy,
            Metric.BinaryAccuracy,
            Metric.BinaryCrossentropy,
            Metric.CategoricalAccuracy,
            Metric.CategoricalCrossentropy,
            Metric.CategoricalHinge,
            Metric.CosineSimilarity,
            Metric.FalseNegatives,
            Metric.FalsePositives,
            Metric.Hinge,
            Metric.KLDivergence,
            Metric.LogCoshError,
            Metric.MeanAbsoluteError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanIoU,
            Metric.MeanSquaredError,
            Metric.MeanSquaredLogarithmicError,
            Metric.Poisson,
            Metric.Precision,
            Metric.Recall,
            Metric.RootMeanSquaredError,
            Metric.SquaredHinge,
            Metric.TopKCategoricalAccuracy,
            Metric.TrueNegatives,
            Metric.TruePositives,
            Metric.SparseCategoricalAccuracy,
            Metric.SparseCategoricalCrossentropy,
            Metric.SparseTopKCategoricalAccuracy
        ],
    },
    {
        "task": Task.Segmentation,
        "losses": [
            Loss.BinaryCrossentropy,
            Loss.CategoricalCrossentropy,
            Loss.CategoricalHinge,
            Loss.CosineSimilarity,
            Loss.Hinge,
            Loss.Huber,
            Loss.LogCosh,
            Loss.KLDivergence,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredError,
            Loss.MeanSquaredLogarithmicError,
            Loss.Poisson,
            Loss.SquaredHinge,
            Loss.SparseCategoricalCrossentropy,
        ],
        "metrics": [
            Metric.DiceCoef,
            Metric.AUC,
            Metric.Accuracy,
            Metric.BinaryAccuracy,
            Metric.BinaryCrossentropy,
            Metric.CategoricalAccuracy,
            Metric.CategoricalCrossentropy,
            Metric.CategoricalHinge,
            Metric.CosineSimilarity,
            Metric.FalseNegatives,
            Metric.FalsePositives,
            Metric.Hinge,
            Metric.KLDivergence,
            Metric.LogCoshError,
            Metric.MeanAbsoluteError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanIoU,
            Metric.MeanSquaredError,
            Metric.MeanSquaredLogarithmicError,
            Metric.Poisson,
            Metric.Precision,
            Metric.Recall,
            Metric.RootMeanSquaredError,
            Metric.SquaredHinge,
            Metric.TrueNegatives,
            Metric.TruePositives,
            Metric.SparseCategoricalAccuracy,
            Metric.SparseCategoricalCrossentropy,
            Metric.SparseTopKCategoricalAccuracy,
        ],
    },
    {
        "task": Task.Regression,
        "losses": [
            Loss.MeanSquaredError,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredLogarithmicError,
            Loss.LogCosh,
            Loss.CosineSimilarity,
            Loss.Hinge,
            Loss.KLDivergence,
            Loss.SquaredHinge
        ],
        "metrics": [
            # Metric.Accuracy,
            Metric.MeanAbsoluteError,
            Metric.MeanSquaredError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanSquaredLogarithmicError,
            Metric.LogCoshError,
            Metric.CosineSimilarity,
            Metric.Hinge,
            Metric.KLDivergence,
            Metric.RootMeanSquaredError,
            Metric.SquaredHinge
        ],
    },
    {
        "task": Task.Timeseries,
        "losses": [
            Loss.CosineSimilarity,
            Loss.Hinge,
            Loss.Huber,
            Loss.KLDivergence,
            Loss.LogCosh,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredError,
            Loss.MeanAbsolutePercentageError,
            Loss.SquaredHinge
        ],
        "metrics": [
            Metric.CosineSimilarity,
            Metric.Hinge,
            Metric.KLDivergence,
            Metric.LogCoshError,
            Metric.MeanAbsoluteError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanSquaredError,
            Metric.MeanSquaredLogarithmicError,
            Metric.RootMeanSquaredError,
            Metric.SquaredHinge
        ],
    },
]
