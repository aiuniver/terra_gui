"""
Предустановки обучения
"""

from enum import Enum


class Task(str, Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
    ObjectDetection = "ObjectDetection"
    Dataframe = "Dataframe"
    TextSegmentation = "TextSegmentation"
    TimeseriesTrend = "TimeseriesTrend"


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
    # SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    SquaredHinge = "SquaredHinge"
    YoloLoss = "YoloLoss"


class Metric(str, Enum):
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
    RecallPercent = "RecallPercent"
    RootMeanSquaredError = "RootMeanSquaredError"
    SquaredHinge = "SquaredHinge"
    TopKCategoricalAccuracy = "TopKCategoricalAccuracy"
    TrueNegatives = "TrueNegatives"
    TruePositives = "TruePositives"
    # SparseCategoricalAccuracy = "SparseCategoricalAccuracy"
    # SparseCategoricalCrossentropy = "SparseCategoricalCrossentropy"
    # SparseTopKCategoricalAccuracy = "SparseTopKCategoricalAccuracy"
    DiceCoef = "DiceCoef"
    BalancedDiceCoef = "BalancedDiceCoef"
    UnscaledMAE = "UnscaledMAE"
    mAP50 = "mAP50"
    mAP95 = "mAP95"


TasksGroups = [
    {
        "task": Task.Classification,
        "losses": [
            Loss.CategoricalCrossentropy,
            Loss.BinaryCrossentropy,
            Loss.CategoricalHinge,
            # Loss.CosineSimilarity,
            # Loss.Hinge,
            # Loss.Huber,
            # Loss.KLDivergence,
            # Loss.LogCosh,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredError,
            Loss.MeanSquaredLogarithmicError,
            # Loss.Poisson,
            # Loss.SparseCategoricalCrossentropy,
            # Loss.SquaredHinge,
        ],
        "metrics": [
            Metric.BalancedRecall,
            # Metric.RecallPercent,
            Metric.BalancedPrecision,
            Metric.BalancedFScore,
            Metric.FScore,
            Metric.CategoricalAccuracy,
            Metric.BinaryAccuracy,
            Metric.CategoricalCrossentropy,
            Metric.BinaryCrossentropy,
            Metric.AUC,
            # Metric.Accuracy,
            # Metric.CategoricalHinge,
            # Metric.CosineSimilarity,
            # Metric.FalseNegatives,
            # Metric.FalsePositives,
            # Metric.Hinge,
            # Metric.KLDivergence,
            # Metric.LogCoshError,
            Metric.MeanAbsoluteError,
            Metric.MeanAbsolutePercentageError,
            # Metric.MeanIoU,
            Metric.MeanSquaredError,
            Metric.MeanSquaredLogarithmicError,
            # Metric.Poisson,
            Metric.Precision,
            Metric.Recall,
            Metric.RootMeanSquaredError,
            # Metric.SquaredHinge,
            # Metric.TopKCategoricalAccuracy,
            # Metric.TrueNegatives,
            # Metric.TruePositives,
            # Metric.SparseCategoricalAccuracy,
            # Metric.SparseCategoricalCrossentropy,
            # Metric.SparseTopKCategoricalAccuracy
        ],
    },
    {
        "task": Task.Segmentation,
        "losses": [
            Loss.CategoricalCrossentropy,
            Loss.BinaryCrossentropy,
            Loss.MeanAbsoluteError,
            # Loss.CategoricalHinge,
            Loss.MeanSquaredError,
            # Loss.CosineSimilarity,
            # Loss.Hinge,
            # Loss.Huber,
            # Loss.LogCosh,
            # Loss.KLDivergence,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredLogarithmicError,
            # Loss.Poisson,
            # Loss.SquaredHinge,
            # Loss.SparseCategoricalCrossentropy,
        ],
        "metrics": [
            Metric.BalancedDiceCoef,
            Metric.DiceCoef,
            Metric.MeanIoU,
            Metric.AUC,
            Metric.BinaryAccuracy,
            Metric.MeanAbsoluteError,
            Metric.MeanSquaredError,
            Metric.RootMeanSquaredError,
            Metric.CategoricalAccuracy,
            Metric.CategoricalCrossentropy,
            # Metric.Accuracy,
            Metric.BinaryCrossentropy,
            # Metric.CategoricalHinge,
            # Metric.CosineSimilarity,
            # Metric.FalseNegatives,
            # Metric.FalsePositives,
            # Metric.Hinge,
            # Metric.KLDivergence,
            # Metric.LogCoshError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanSquaredLogarithmicError,
            # Metric.Poisson,
            Metric.Precision,
            Metric.Recall,
            # Metric.SquaredHinge,
            # Metric.TrueNegatives,
            # Metric.TruePositives,
            # Metric.SparseCategoricalAccuracy,
            # Metric.SparseCategoricalCrossentropy,
            # Metric.SparseTopKCategoricalAccuracy,
        ],
    },
    {
        "task": Task.Regression,
        "losses": [
            Loss.MeanSquaredError,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredLogarithmicError,
            # Loss.LogCosh,
            # Loss.CosineSimilarity,
            # Loss.Hinge,
            # Loss.KLDivergence,
            # Loss.SquaredHinge,
        ],
        "metrics": [
            Metric.UnscaledMAE,
            Metric.MeanAbsoluteError,
            Metric.MeanSquaredError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanSquaredLogarithmicError,
            # Metric.LogCoshError,
            # Metric.CosineSimilarity,
            # Metric.Hinge,
            # Metric.KLDivergence,
            Metric.RootMeanSquaredError,
            # Metric.SquaredHinge,
        ],
    },
    {
        "task": Task.Timeseries,
        "losses": [
            Loss.MeanSquaredError,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredLogarithmicError,
            # Loss.CosineSimilarity,
            # Loss.Hinge,
            # Loss.Huber,
            # Loss.KLDivergence,
            # Loss.LogCosh,
            # Loss.SquaredHinge,
        ],
        "metrics": [
            Metric.UnscaledMAE,
            Metric.MeanAbsoluteError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanSquaredError,
            Metric.MeanSquaredLogarithmicError,
            Metric.RootMeanSquaredError,
            # Metric.CosineSimilarity,
            # Metric.Hinge,
            # Metric.KLDivergence,
            # Metric.LogCoshError,
            # Metric.SquaredHinge,
        ],
    },
    {
        "task": Task.ObjectDetection,
        "losses": [Loss.YoloLoss],
        "metrics": [
            Metric.mAP50,
            # Metric.mAP95
        ],
    },
    {
        "task": Task.TimeseriesTrend,
        "losses": [
            Loss.CategoricalCrossentropy,
            Loss.BinaryCrossentropy,
            Loss.CategoricalHinge,
            # Loss.CosineSimilarity,
            # Loss.Hinge,
            # Loss.Huber,
            # Loss.KLDivergence,
            # Loss.LogCosh,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredError,
            Loss.MeanSquaredLogarithmicError,
            # Loss.Poisson,
            # Loss.SparseCategoricalCrossentropy,
            # Loss.SquaredHinge,
        ],
        "metrics": [
            Metric.BalancedRecall,
            # Metric.RecallPercent,
            Metric.BalancedPrecision,
            Metric.BalancedFScore,
            Metric.FScore,
            Metric.CategoricalAccuracy,
            Metric.BinaryAccuracy,
            Metric.CategoricalCrossentropy,
            Metric.BinaryCrossentropy,
            Metric.AUC,
            # Metric.Accuracy,
            # Metric.CategoricalHinge,
            # Metric.CosineSimilarity,
            # Metric.FalseNegatives,
            # Metric.FalsePositives,
            # Metric.Hinge,
            # Metric.KLDivergence,
            # Metric.LogCoshError,
            Metric.MeanAbsoluteError,
            Metric.MeanAbsolutePercentageError,
            # Metric.MeanIoU,
            Metric.MeanSquaredError,
            Metric.MeanSquaredLogarithmicError,
            # Metric.Poisson,
            Metric.Precision,
            Metric.Recall,
            Metric.RootMeanSquaredError,
            # Metric.SquaredHinge,
            # Metric.TopKCategoricalAccuracy,
            # Metric.TrueNegatives,
            # Metric.TruePositives,
            # Metric.SparseCategoricalAccuracy,
            # Metric.SparseCategoricalCrossentropy,
            # Metric.SparseTopKCategoricalAccuracy
        ],
    },
    {
        "task": Task.TextSegmentation,
        "losses": [
            Loss.BinaryCrossentropy,
            Loss.MeanAbsoluteError,
            Loss.CategoricalCrossentropy,
            Loss.CategoricalHinge,
            Loss.MeanSquaredError,
            # Loss.CosineSimilarity,
            # Loss.Hinge,
            # Loss.Huber,
            # Loss.LogCosh,
            # Loss.KLDivergence,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredLogarithmicError,
            # Loss.Poisson,
            # Loss.SquaredHinge,
            # Loss.SparseCategoricalCrossentropy,
        ],
        "metrics": [
            Metric.BalancedDiceCoef,
            Metric.DiceCoef,
            Metric.MeanIoU,
            Metric.BinaryAccuracy,
            Metric.AUC,
            Metric.MeanAbsoluteError,
            Metric.MeanSquaredError,
            Metric.RootMeanSquaredError,
            Metric.CategoricalAccuracy,
            Metric.CategoricalCrossentropy,
            # Metric.Accuracy,
            Metric.BinaryCrossentropy,
            # Metric.CategoricalHinge,
            # Metric.CosineSimilarity,
            # Metric.FalseNegatives,
            # Metric.FalsePositives,
            # Metric.Hinge,
            # Metric.KLDivergence,
            # Metric.LogCoshError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanSquaredLogarithmicError,
            # Metric.Poisson,
            # Metric.Precision,
            # Metric.Recall,
            # Metric.SquaredHinge,
            # Metric.TrueNegatives,
            # Metric.TruePositives,
            # Metric.SparseCategoricalAccuracy,
            # Metric.SparseCategoricalCrossentropy,
            # Metric.SparseTopKCategoricalAccuracy,
        ],
    },
    {
        "task": Task.Dataframe,
        "losses": [
            Loss.MeanSquaredError,
            Loss.MeanAbsoluteError,
            Loss.MeanAbsolutePercentageError,
            Loss.MeanSquaredLogarithmicError,
            # Loss.CosineSimilarity,
            # Loss.Hinge,
            # Loss.Huber,
            # Loss.KLDivergence,
            # Loss.LogCosh,
            # Loss.SquaredHinge,
        ],
        "metrics": [
            # Metric.UnscaledMAE,
            Metric.MeanAbsoluteError,
            Metric.MeanAbsolutePercentageError,
            Metric.MeanSquaredError,
            Metric.MeanSquaredLogarithmicError,
            Metric.RootMeanSquaredError,
            # Metric.CosineSimilarity,
            # Metric.Hinge,
            # Metric.KLDivergence,
            # Metric.LogCoshError,
            # Metric.SquaredHinge,
        ],
    },
]
