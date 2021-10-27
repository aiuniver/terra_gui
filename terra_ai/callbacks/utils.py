import math

import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.data.training.extra import BalanceSortedChoice

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
        auto_corr_true.append(get_auto_corr(y_true[:-(i + 1)], y_true[(i + 1):]))

    auto_corr_pred = []
    for i in range(depth):
        auto_corr_pred.append(get_auto_corr(y_true[:-(i + 1)], y_pred[(i + 1):]))

    x_axis = np.arange(depth).astype('int').tolist()
    return x_axis, auto_corr_true, auto_corr_pred


def round_list(x: list) -> list:
    update_x = []
    for data in x:
        if data > 1:
            update_x.append(np.round(data, -int(math.floor(math.log10(abs(data))) - 3)).item())
        else:
            update_x.append(np.round(data, -int(math.floor(math.log10(abs(data))) - 2)).item())
    return update_x


def sort_dict(dict_to_sort: dict, mode: BalanceSortedChoice = BalanceSortedChoice.alphabetic):
    if mode == BalanceSortedChoice.alphabetic:
        sorted_keys = sorted(dict_to_sort)
        sorted_values = []
        for w in sorted_keys:
            sorted_values.append(dict_to_sort[w])
        return tuple(sorted_keys), tuple(sorted_values)
    elif mode == BalanceSortedChoice.ascending:
        sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get)
        sorted_values = []
        for w in sorted_keys:
            sorted_values.append(dict_to_sort[w])
        return tuple(sorted_keys), tuple(sorted_values)
    elif mode == BalanceSortedChoice.descending:
        sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get, reverse=True)
        sorted_values = []
        for w in sorted_keys:
            sorted_values.append(dict_to_sort[w])
        return tuple(sorted_keys), tuple(sorted_values)
    else:
        return tuple(dict_to_sort.keys()), tuple(dict_to_sort.values())


def dice_coef(y_true, y_pred, batch_mode=True, smooth=1.0):
    axis = tuple(np.arange(1, len(y_true.shape))) if batch_mode else None
    intersection = np.sum(y_true * y_pred, axis=axis)
    union = np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis)
    return (2.0 * intersection + smooth) / (union + smooth)


def get_confusion_matrix(y_true, y_pred, get_percent=True) -> tuple:
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = None
    if get_percent:
        cm_percent = np.zeros_like(cm).astype('float')
        for i in range(len(cm)):
            total = np.sum(cm[i])
            for j in range(len(cm[i])):
                cm_percent[i][j] = round(cm[i][j] * 100 / total, 1)
    return cm.astype('float').tolist(), cm_percent.astype('float').tolist()


def get_classification_report(y_true, y_pred, labels):
    cr = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    return_stat = []
    for lbl in labels:
        return_stat.append(
            {
                'Класс': lbl,
                "Точность": round(float(cr.get(lbl).get('precision')) * 100, 2),
                "Чувствительность": round(float(cr.get(lbl).get('recall')) * 100, 2),
                "F1-мера": round(float(cr.get(lbl).get('f1-score')) * 100, 2),
                "Количество": int(cr.get(lbl).get('support'))
            }
        )
    for i in ['macro avg', 'micro avg', 'samples avg', 'weighted avg']:
        return_stat.append(
            {
                'Класс': i,
                "Точность": round(float(cr.get(i).get('precision')) * 100, 2),
                "Чувствительность": round(float(cr.get(i).get('recall')) * 100, 2),
                "F1-мера": round(float(cr.get(i).get('f1-score')) * 100, 2),
                "Количество": int(cr.get(i).get('support'))
            }
        )
    return return_stat


def get_error_distribution(y_true, y_pred, absolute=True):
    error = (y_true - y_pred)  # "* 100 / y_true
    if absolute:
        error = np.abs(error)
    return get_distribution_histogram(error, categorical=False)


def get_time_series_graphic(data):
    return np.arange(len(data)).astype('int').tolist(), np.array(data).astype('float').tolist()


def get_correlation_matrix(data_frame: DataFrame):
    corr = data_frame.corr()
    labels = []
    for lbl in list(corr.columns):
        labels.append(lbl.split("_", 1)[-1])
    return labels, np.array(np.round(corr, 2)).astype('float').tolist()


def get_scatter(y_true, y_pred):
    return clean_data_series([y_true, y_pred], mode="duo")


def get_distribution_histogram(data_series, categorical=True):
    if categorical:
        hist_data = pd.Series(data_series).value_counts()
        return hist_data.index.to_list(), hist_data.to_list()
    else:
        bins = int(len(data_series) / 10)
        data_series = clean_data_series([data_series], mode="mono")
        bar_values, x_labels = np.histogram(data_series, bins=bins)
        new_x = []
        for i in range(len(x_labels[:-1])):
            new_x.append(np.mean([x_labels[i], x_labels[i + 1]]))
        new_x = np.array(new_x)
        return new_x.astype('float').tolist(), bar_values.astype('int').tolist()


def clean_data_series(data_series: list, mode="mono"):
    if mode == "mono":
        sort_x = pd.Series(data_series[0])
        sort_x = sort_x[sort_x > sort_x.quantile(0.02)]
        sort_x = sort_x[sort_x < sort_x.quantile(0.98)]
        data_series = np.array(sort_x)
        return data_series
    elif mode == "duo":
        sort = pd.DataFrame({
            'y_true': np.array(data_series[0]).squeeze(),
            'y_pred': np.array(data_series[1]).squeeze(),
        })
        sort = sort[sort['y_true'] > sort['y_true'].quantile(0.05)]
        sort = sort[sort['y_true'] < sort['y_true'].quantile(0.95)]
        return sort['y_true'].to_list(), sort['y_pred'].to_list()
    else:
        return None


def get_image_class_colormap(array: np.ndarray, colors: list, class_id: int, save_path: str):
    array = np.expand_dims(np.argmax(array, axis=-1), axis=-1) * 512
    array = np.where(
        array == class_id * 512,
        np.array(colors[class_id]) if np.sum(np.array(colors[class_id])) > 50 else np.array((255, 255, 255)),
        np.array((0, 0, 0))
    )
    array = (np.sum(array, axis=0) / len(array)).astype("uint8")
    matplotlib.image.imsave(save_path, array)


def round_loss_metric(x: float) -> float:
    if not x:
        return x
    elif x > 1000:
        return np.round(x, 0).item()
    elif x > 1:
        return np.round(x, -int(math.floor(math.log10(abs(x))) - 3)).item()
    else:
        return np.round(x, -int(math.floor(math.log10(abs(x))) - 2)).item()


def fill_graph_plot_data(x: list, y: list, label=None):
    return {'label': label, 'x': x, 'y': y}


def fill_graph_front_structure(_id: int, _type: str, graph_name: str, short_name: str,
                                x_label: str, y_label: str, plot_data: list, best: list = None,
                                type_data: str = None, progress_state: str = None):
    return {
        'id': _id,
        'type': _type,
        'type_data': type_data,
        'graph_name': graph_name,
        'short_name': short_name,
        'x_label': x_label,
        'y_label': y_label,
        'plot_data': plot_data,
        'best': best,
        'progress_state': progress_state
    }


def fill_heatmap_front_structure(_id: int, _type: str, graph_name: str, short_name: str,
                                  x_label: str, y_label: str, labels: list, data_array: list,
                                  type_data: str = None, data_percent_array: list = None,
                                  progress_state: str = None):
    return {
        'id': _id,
        'type': _type,
        'type_data': type_data,
        'graph_name': graph_name,
        'short_name': short_name,
        'x_label': x_label,
        'y_label': y_label,
        'labels': labels,
        'data_array': data_array,
        'data_percent_array': data_percent_array,
        'progress_state': progress_state
    }


def fill_table_front_structure(_id: int, graph_name: str, plot_data: list):
    return {'id': _id, 'type': 'table', 'graph_name': graph_name, 'plot_data': plot_data}


def get_y_true(options, output_id):
    if not options.data.use_generator:
        y_true = options.Y.get('val').get(f"{output_id}")
    else:
        y_true = []
        for _, y_val in options.dataset['val'].batch(1):
            y_true.extend(y_val.get(f'{output_id}').numpy())
        y_true = np.array(y_true)
    return y_true

