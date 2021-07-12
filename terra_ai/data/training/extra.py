"""
## Дополнительные структуры данных для оптимайзеров
"""

from enum import Enum


class ArchitectureChoice(str, Enum):
    Basic = "Базовая"
    Yolo = "Yolo"


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
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"


class LossChoice(str, Enum):
    categorical_crossentropy = "Categorical crossentropy"
    binary_crossentropy = "Binary crossentropy"
    mse = "MSE"
    squared_hinge = "Squared hinge"
    hinge = "hinge"
    categorical_hinge = "Categorical hinge"
    sparse_categorical_crossentropy = "Sparse categorical crossentropy"
    kl_divergence = "Kullback-Leibler divergence"
    poisson = "Poisson"
    mae = "MAE"
    mape = "Mape"
    msle = "MSLE"
    log_cosh = "Log cosh"
    cosine_similarity = "Cosine similarity"


class MetricChoice(str, Enum):
    accuracy = "Accuracy"
    binary_accuracy = "Binary accuracy"
    binary_crossentropy = "Binary crossentropy"
    categorical_accuracy = "Categorical accuracy"
    categorical_crossentropy = "Categorical crossentropy"
    sparse_categorical_accuracy = "Sparse categorical accuracy"
    sparse_categorical_crossentropy = "Sparse categorical crossentropy"
    top_k_categorical_accuracy = "Top K categorical accuracy"
    sparse_top_k_categorical_accuracy = "Sparse top K categorical accuracy"
    hinge = "Hinge"
    kullback_leibler_divergence = "Kullback-Leibler divergence"
    poisson = "Poisson"
    dice_coef = "Dice coef"
    mean_io_u = "Mean IO U"
    mae = "MAE"
    mse = "MSE"
    mape = "Mape"
    msle = "MSLE"
    logcosh = "Log cosh"
    cosine_similarity = "Cosine similarity"
