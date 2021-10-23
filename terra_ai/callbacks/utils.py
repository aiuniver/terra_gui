import numpy as np


loss_metric_config = {
    "loss": {
        "BinaryCrossentropy": {
            "log_name": "binary_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.losses",
        },
        "CategoricalCrossentropy": {
            "log_name": "categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "CategoricalHinge": {
            "log_name": "categorical_hinge",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        'ContrastiveLoss': {
            "log_name": "contrastive_loss",
            "mode": "min",
            "module": "tensorflow_addons.losses"
        },
        "CosineSimilarity": {
            "log_name": "cosine_similarity",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },  # min if loss, max if metric
        "Hinge": {
            "log_name": "hinge",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "Huber": {
            "log_name": "huber",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "KLDivergence": {
            "log_name": "kullback_leibler_divergence",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "LogCosh": {
            "log_name": "logcosh",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanAbsoluteError": {
            "log_name": "mean_absolute_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanAbsolutePercentageError": {
            "log_name": "mean_absolute_percentage_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanSquaredError": {
            "log_name": "mean_squared_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanSquaredLogarithmicError": {
            "log_name": "mean_squared_logarithmic_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "Poisson": {
            "log_name": "poisson",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "SparseCategoricalCrossentropy": {
            "log_name": "sparse_categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "SquaredHinge": {
            "log_name": "squared_hinge",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "YoloLoss": {
            "log_name": "yolo_loss",
            "mode": "min",
            "module": "terra_ai.training.customlosses"
        }
    },
    "metric": {
        "AUC": {
            "log_name": "auc",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "Accuracy": {
            "log_name": "accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "BinaryAccuracy": {
            "log_name": "binary_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "BinaryCrossentropy": {
            "log_name": "binary_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "CategoricalAccuracy": {
            "log_name": "categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "CategoricalCrossentropy": {
            "log_name": "categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "CategoricalHinge": {
            "log_name": "categorical_hinge",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "CosineSimilarity": {
            "log_name": "cosine_similarity",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },  # min if loss, max if metric
        "DiceCoef": {
            "log_name": "dice_coef",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "FalseNegatives": {
            "log_name": "false_negatives",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "FalsePositives": {
            "log_name": "false_positives",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "Hinge": {
            "log_name": "hinge",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "KLDivergence": {
            "log_name": "kullback_leibler_divergence",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "LogCoshError": {
            "log_name": "logcosh",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanAbsoluteError": {
            "log_name": "mean_absolute_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanAbsolutePercentageError": {
            "log_name": "mean_absolute_percentage_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanIoU": {
            "log_name": "mean_io_u",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "MeanSquaredError": {
            "log_name": "mean_squared_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanSquaredLogarithmicError": {
            "log_name": "mean_squared_logarithmic_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "Poisson": {
            "log_name": "poisson",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "Precision": {
            "log_name": "precision",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "Recall": {
            "log_name": "recall",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "RecallPercent": {
            "log_name": "recall_percent",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "BalancedRecall": {
            "log_name": "balanced_recall",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "RootMeanSquaredError": {
            "log_name": "root_mean_squared_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "SparseCategoricalAccuracy": {
            "log_name": "sparse_categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "SparseCategoricalCrossentropy": {
            "log_name": "sparse_categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "SparseTopKCategoricalAccuracy": {
            "log_name": "sparse_top_k_categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "SquaredHinge": {
            "log_name": "squared_hinge",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "TopKCategoricalAccuracy": {
            "log_name": "top_k_categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "TrueNegatives": {
            "log_name": "true_negatives",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "TruePositives": {
            "log_name": "true_positives",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "UnscaledMAE": {
            "log_name": "unscaled_mae",
            "mode": "min",
            "module": "terra_ai.training.customlosses"
        },
    }
}


def class_counter(y_array, classes_names: list, ohe=True):
    """
    class_dict = {
        "class_name": int
    }
    """
    class_dict = {}
    for cl in classes_names:
        class_dict[cl] = 0
    y_array = np.argmax(y_array, axis=-1) if ohe else np.squeeze(y_array)
    for y in y_array:
        class_dict[classes_names[y]] += 1
    return class_dict


def get_autocorrelation_graphic(y_true, y_pred, depth=10) -> (list, list, list):

    def get_auto_corr(a, b):
        ma = a.mean()
        mb = b.mean()
        mab = (a * b).mean()
        sa = a.std()
        sb = b.std()

        val = 1
        if sa > 0 and sb > 0:
            val = (mab - ma * mb) / (sa * sb)
        return val

    auto_corr_true = []
    for i in range(depth):
        auto_corr_true.append(get_auto_corr(y_true[:-(i+1)], y_true[(i+1):]))

    auto_corr_pred = []
    for i in range(depth):
        auto_corr_pred.append(get_auto_corr(y_true[:-(i+1)], y_pred[(i+1):]))

    x_axis = np.arange(depth).astype('int').tolist()
    return x_axis, auto_corr_true, auto_corr_pred
