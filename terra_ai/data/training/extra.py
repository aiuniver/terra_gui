"""
## Дополнительные структуры данных для оптимайзеров
"""

from enum import Enum


class CheckpointIndicatorChoice(str, Enum):
    val = "val"
    train = "train"


class CheckpointModeChoice(str, Enum):
    min = "min"
    max = "max"


class CheckpointTypeChoice(str, Enum):
    metrics = "metrics"
    loss = "loss"


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
    classification = "classification"
    segmentation = "segmentation"
    regression = "regression"
    timeseries = "timeseries"


class LossChoice(str, Enum):
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


class MetricChoice(str, Enum):
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
