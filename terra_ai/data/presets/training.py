"""
Предустановки обучения
"""

from enum import Enum


class Task(str, Enum):
    classification = "classification"
    segmentation = "segmentation"
    regression = "regression"
    timeseries = "timeseries"


class Loss(str, Enum):
    categorical_crossentropy = "categorical_crossentropy"
    binary_crossentropy = "binary_crossentropy"
    mse = "mse"
    squared_hinge = "squared_hinge"
    hinge = "hinge"
    categorical_hinge = "categorical_hinge"
    sparse_categorical_crossentropy = "sparse_categorical_crossentropy"
    kl_divergence = "kl_divergence"
    poisson = "poisson"
    mae = "mae"
    mape = "mape"
    msle = "msle"
    log_cosh = "log_cosh"
    cosine_similarity = "cosine_similarity"


class Metric(str, Enum):
    accuracy = "accuracy"
    binary_accuracy = "binary_accuracy"
    binary_crossentropy = "binary_crossentropy"
    categorical_accuracy = "categorical_accuracy"
    categorical_crossentropy = "categorical_crossentropy"
    sparse_categorical_accuracy = "sparse_categorical_accuracy"
    sparse_categorical_crossentropy = "sparse_categorical_crossentropy"
    top_k_categorical_accuracy = "top_k_categorical_accuracy"
    sparse_top_k_categorical_accuracy = "sparse_top_k_categorical_accuracy"
    hinge = "hinge"
    kullback_leibler_divergence = "kullback_leibler_divergence"
    poisson = "poisson"
    dice_coef = "dice_coef"
    mean_io_u = "mean_io_u"
    mae = "mae"
    mse = "mse"
    mape = "mape"
    msle = "msle"
    logcosh = "logcosh"
    cosine_similarity = "cosine_similarity"


TasksGroups = [
    {
        "alias": Task.classification,
        "losses": [
            Loss.categorical_crossentropy,
            Loss.binary_crossentropy,
            Loss.mse,
            Loss.squared_hinge,
            Loss.hinge,
            Loss.categorical_hinge,
            Loss.sparse_categorical_crossentropy,
            Loss.kl_divergence,
            Loss.poisson,
        ],
        "metrics": [
            Metric.accuracy,
            Metric.binary_accuracy,
            Metric.binary_crossentropy,
            Metric.categorical_accuracy,
            Metric.categorical_crossentropy,
            Metric.sparse_categorical_accuracy,
            Metric.sparse_categorical_crossentropy,
            Metric.top_k_categorical_accuracy,
            Metric.sparse_top_k_categorical_accuracy,
            Metric.hinge,
            Metric.kullback_leibler_divergence,
            Metric.poisson,
        ],
    },
    {
        "alias": Task.segmentation,
        "losses": [
            Loss.categorical_crossentropy,
            Loss.binary_crossentropy,
            Loss.squared_hinge,
            Loss.hinge,
            Loss.categorical_hinge,
            Loss.sparse_categorical_crossentropy,
            Loss.kl_divergence,
            Loss.poisson,
        ],
        "metrics": [
            Metric.dice_coef,
            Metric.mean_io_u,
            Metric.accuracy,
            Metric.binary_accuracy,
            Metric.binary_crossentropy,
            Metric.categorical_accuracy,
            Metric.categorical_crossentropy,
            Metric.sparse_categorical_accuracy,
            Metric.sparse_categorical_crossentropy,
            Metric.top_k_categorical_accuracy,
            Metric.sparse_top_k_categorical_accuracy,
            Metric.hinge,
            Metric.kullback_leibler_divergence,
            Metric.poisson,
        ],
    },
    {
        "alias": Task.regression,
        "losses": [
            Loss.mse,
            Loss.mae,
            Loss.mape,
            Loss.msle,
            Loss.log_cosh,
            Loss.cosine_similarity,
        ],
        "metrics": [
            Metric.accuracy,
            Metric.mae,
            Metric.mse,
            Metric.mape,
            Metric.msle,
            Metric.logcosh,
            Metric.cosine_similarity,
        ],
    },
    {
        "alias": Task.timeseries,
        "losses": [
            Loss.mse,
            Loss.mae,
            Loss.mape,
            Loss.msle,
            Loss.log_cosh,
            Loss.cosine_similarity,
        ],
        "metrics": [
            Metric.accuracy,
            Metric.mae,
            Metric.mse,
            Metric.mape,
            Metric.msle,
            Metric.logcosh,
            Metric.cosine_similarity,
        ],
    },
]
