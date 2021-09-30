from typing import Tuple, List

from terra_ai.data.modeling.extra import LayerGroupChoice
from .extra import (
    TaskChoice,
    ArchitectureChoice,
    OptimizerChoice,
    LossChoice,
    MetricChoice,
    ShowImagesChoice,
)
from .architectures import Basic


LossesData = {
    TaskChoice.Classification.value: [
        LossChoice.CategoricalCrossentropy.value,
        LossChoice.BinaryCrossentropy.value,
        LossChoice.CategoricalHinge.value,
        LossChoice.SquaredHinge.value,
        LossChoice.Hinge.value,
        # LossChoice.SparseCategoricalCrossentropy.value,
        LossChoice.KLDivergence.value,
        LossChoice.Poisson.value,
        LossChoice.Huber.value,
        LossChoice.LogCosh.value,
        LossChoice.MeanAbsoluteError.value,
        LossChoice.MeanAbsolutePercentageError.value,
        LossChoice.MeanSquaredError.value,
        LossChoice.MeanSquaredLogarithmicError.value
    ],
    TaskChoice.Segmentation.value: [
        LossChoice.MeanAbsoluteError.value,
        LossChoice.MeanSquaredError.value,
        LossChoice.CategoricalCrossentropy.value,
        LossChoice.BinaryCrossentropy.value,
        LossChoice.SquaredHinge.value,
        LossChoice.Hinge.value,
        LossChoice.CategoricalHinge.value,
        # LossChoice.SparseCategoricalCrossentropy.value,
        LossChoice.KLDivergence.value,
        LossChoice.Poisson.value,
        LossChoice.CosineSimilarity.value,
        LossChoice.Huber.value,
        LossChoice.LogCosh.value,
        LossChoice.MeanAbsolutePercentageError.value,
        LossChoice.MeanSquaredLogarithmicError.value
    ],
    TaskChoice.Regression.value: [
        LossChoice.MeanSquaredError.value,
        LossChoice.MeanAbsoluteError.value,
        LossChoice.MeanAbsolutePercentageError.value,
        LossChoice.MeanSquaredLogarithmicError.value,
        LossChoice.LogCosh.value,
        LossChoice.CosineSimilarity.value,
        LossChoice.Hinge.value,
        LossChoice.KLDivergence.value,
        LossChoice.SquaredHinge.value
    ],
    TaskChoice.Timeseries.value: [
        LossChoice.MeanSquaredError.value,
        LossChoice.MeanAbsoluteError.value,
        LossChoice.MeanAbsolutePercentageError.value,
        LossChoice.MeanSquaredLogarithmicError.value,
        LossChoice.LogCosh.value,
        LossChoice.CosineSimilarity.value,
        LossChoice.Hinge.value,
        LossChoice.Huber.value,
        LossChoice.KLDivergence.value,
        LossChoice.SquaredHinge.value
    ],
}


MetricsData = {
    TaskChoice.Classification.value: [
        MetricChoice.CategoricalAccuracy.value,
        MetricChoice.CategoricalCrossentropy.value,
        MetricChoice.BinaryAccuracy.value,
        MetricChoice.BinaryCrossentropy.value,
        MetricChoice.CategoricalHinge.value,
        MetricChoice.Accuracy.value,
        # MetricChoice.SparseCategoricalAccuracy.value,
        # MetricChoice.SparseCategoricalCrossentropy.value,
        MetricChoice.TopKCategoricalAccuracy.value,
        # MetricChoice.SparseTopKCategoricalAccuracy.value,
        MetricChoice.Hinge.value,
        MetricChoice.KullbackLeiblerDivergence.value,
        MetricChoice.Poisson.value,
        MetricChoice.AUC.value,
        MetricChoice.CosineSimilarity.value,
        MetricChoice.FalseNegatives.value,
        MetricChoice.FalsePositives.value,
        MetricChoice.KLDivergence.value,
        MetricChoice.LogCoshError.value,
        MetricChoice.MeanAbsoluteError.value,
        MetricChoice.MeanAbsolutePercentageError.value,
        MetricChoice.MeanSquaredError.value,
        MetricChoice.MeanSquaredLogarithmicError.value,
        MetricChoice.MeanIoU.value,
        MetricChoice.Precision.value,
        MetricChoice.Recall.value,
        MetricChoice.RootMeanSquaredError.value,
        MetricChoice.SquaredHinge.value,
        MetricChoice.TrueNegatives.value,
        MetricChoice.TruePositives.value,
    ],
    TaskChoice.Segmentation.value: [
        MetricChoice.DiceCoef.value,
        MetricChoice.MeanAbsoluteError.value,
        MetricChoice.MeanAbsolutePercentageError.value,
        MetricChoice.MeanSquaredError.value,
        MetricChoice.MeanSquaredLogarithmicError.value,
        MetricChoice.CategoricalAccuracy.value,
        MetricChoice.CategoricalCrossentropy.value,
        MetricChoice.MeanIoU.value,
        MetricChoice.Accuracy.value,
        MetricChoice.BinaryAccuracy.value,
        MetricChoice.BinaryCrossentropy.value,
        # MetricChoice.SparseCategoricalAccuracy.value,
        # MetricChoice.SparseCategoricalCrossentropy.value,
        # MetricChoice.SparseTopKCategoricalAccuracy.value,
        MetricChoice.Hinge.value,
        MetricChoice.KLDivergence.value,
        MetricChoice.Poisson.value,
        MetricChoice.AUC.value,
        MetricChoice.CategoricalHinge.value,
        MetricChoice.CosineSimilarity.value,
        MetricChoice.FalseNegatives.value,
        MetricChoice.FalsePositives.value,
        MetricChoice.LogCoshError.value,
        MetricChoice.Precision.value,
        MetricChoice.Recall.value,
        MetricChoice.RootMeanSquaredError.value,
        MetricChoice.SquaredHinge.value,
        MetricChoice.TrueNegatives.value,
        MetricChoice.TruePositives.value,
    ],
    TaskChoice.Regression.value: [
        MetricChoice.MeanAbsoluteError.value,
        MetricChoice.MeanSquaredError.value,
        MetricChoice.MeanAbsolutePercentageError.value,
        MetricChoice.MeanSquaredLogarithmicError.value,
        MetricChoice.Accuracy.value,
        MetricChoice.LogCoshError.value,
        MetricChoice.CosineSimilarity.value,
        MetricChoice.Hinge.value,
        MetricChoice.KLDivergence.value,
        MetricChoice.RootMeanSquaredError.value,
        MetricChoice.SquaredHinge
    ],
    TaskChoice.Timeseries.value: [
        MetricChoice.MeanAbsoluteError.value,
        MetricChoice.MeanSquaredError.value,
        MetricChoice.MeanAbsolutePercentageError.value,
        MetricChoice.MeanSquaredLogarithmicError.value,
        MetricChoice.Accuracy.value,
        MetricChoice.LogCoshError.value,
        MetricChoice.CosineSimilarity.value,
        MetricChoice.Hinge.value,
        MetricChoice.KLDivergence.value,
        MetricChoice.RootMeanSquaredError.value,
        MetricChoice.SquaredHinge
    ],
}


ShowEveryEpochCallback = {
    "type": "bool",
    "label": "Каждую эпоху",
    "default": True,
}
PlotLossMetricCallback = {
    "type": "bool",
    "label": "Loss",
    "default": True,
}
PlotMetricCallback = {
    "type": "bool",
    "label": "Данные метрики",
    "default": True,
}
PlotLossForClassesCallback = {
    "type": "bool",
    "label": "Loss по каждому классу",
    "default": True,
}
PlotMetricForClassesCallback = {
    "type": "bool",
    "label": "Данные метрики по каждому классу",
    "default": True,
}
PlotScatterCallback = {
    "type": "bool",
    "label": "Скаттеры",
    "default": True,
}
PlotAutocorrelationCallback = {
    "type": "bool",
    "label": "График автокорреляции",
    "default": True,
}
PlotPredAndTrueCallback = {
    "type": "bool",
    "label": "Графики предсказания и истинного ряда",
    "default": True,
}
PlotFinalCallback = {
    "type": "bool",
    "label": "Графики в конце",
    "default": True,
}
ShowImagesCallback = {
    "type": "str",
    "label": "Изображения по метрике",
    "default": "Best",
    "list": True,
    "available": [("", "Не выводить")] + ShowImagesChoice.options(),
}


def get_outputs_config(layers) -> list:
    output = []
    for layer in layers.values():
        if layer.config.location_type != LayerGroupChoice.output:
            continue
        layer_data = Basic.LayerData(
            **{
                "name": layer.config.name,
                "alias": layer.config.dts_layer_name,
                "fields": {
                    "task": {
                        "type": "str",
                        "label": "Задача",
                        "default": "",
                    },
                    "loss": {
                        "type": "str",
                        "label": "Loss",
                        "default": "",
                        "list": True,
                        "available": [],
                    },
                    "metrics": {
                        "type": "str",
                        "label": "Метрика",
                        "default": "",
                        "list": True,
                        "available": [],
                    },
                    "classes_quantity": {
                        "type": "int",
                        "label": "Количество классов",
                        "readonly": True,
                        "default": layer.config.num_classes,
                    },
                },
            }
        )
        output.append(layer_data)
    return output


def get_callbacks_config(layers) -> list:
    output = []
    for layer in layers.values():
        if layer.config.location_type != LayerGroupChoice.output:
            continue
        output.append(layer.config.name)
    return output


def get_checkpoint_layer(layers) -> List[Tuple[str, str]]:
    output = []
    for layer in layers.values():
        if layer.config.location_type != LayerGroupChoice.output:
            continue
        output.append((layer.config.dts_layer_name, f"Слой «{layer.config.name}»"))
    return output


data = {
    "optimizers": {
        "SGD": {
            "momentum": {
                "type": "int",
                "label": "Momentum",
                "default": 0,
            },
            "nesterov": {
                "type": "bool",
                "label": "Nesterov",
                "default": False,
            },
        },
        "RMSprop": {
            "rho": {
                "type": "float",
                "label": "RHO",
                "default": 0.9,
            },
            "momentum": {
                "type": "int",
                "label": "Momentum",
                "default": 0,
            },
            "epsilon": {
                "type": "float",
                "label": "Epsilon",
                "default": 1e-7,
            },
            "centered": {
                "type": "bool",
                "label": "Centered",
                "default": False,
            },
        },
        "Adam": {
            "beta_1": {
                "type": "float",
                "label": "Beta 1",
                "default": 0.9,
            },
            "beta_2": {
                "type": "float",
                "label": "Beta 2",
                "default": 0.999,
            },
            "epsilon": {
                "type": "float",
                "label": "Epsilon",
                "default": 1e-7,
            },
            "amsgrad": {
                "type": "bool",
                "label": "Amsgrad",
                "default": False,
            },
        },
        "Adadelta": {
            "rho": {
                "type": "float",
                "label": "RHO",
                "default": 0.95,
            },
            "epsilon": {
                "type": "float",
                "label": "Epsilon",
                "default": 1e-7,
            },
        },
        "Adagrad": {
            "initial_accumulator_value": {
                "type": "float",
                "label": "Initial accumulator value",
                "default": 0.1,
            },
            "epsilon": {
                "type": "float",
                "label": "Epsilon",
                "default": 1e-7,
            },
        },
        "Adamax": {
            "beta_1": {
                "type": "float",
                "label": "Beta 1",
                "default": 0.9,
            },
            "beta_2": {
                "type": "float",
                "label": "Beta 2",
                "default": 0.999,
            },
            "epsilon": {
                "type": "float",
                "label": "Epsilon",
                "default": 1e-7,
            },
        },
        "Nadam": {
            "beta_1": {
                "type": "float",
                "label": "Beta 1",
                "default": 0.9,
            },
            "beta_2": {
                "type": "float",
                "label": "Beta 2",
                "default": 0.999,
            },
            "epsilon": {
                "type": "float",
                "label": "Epsilon",
                "default": 1e-7,
            },
        },
        "Ftrl": {
            "learning_rate_power": {
                "type": "float",
                "label": "Learning rate power",
                "default": -0.5,
            },
            "initial_accumulator_value": {
                "type": "float",
                "label": "Initial accumulator value",
                "default": 0.1,
            },
            "l1_regularization_strength": {
                "type": "int",
                "label": "L1 regularization strength",
                "default": 0,
            },
            "l2_regularization_strength": {
                "type": "int",
                "label": "L2 regularization strength",
                "default": 0,
            },
            "l2_shrinkage_regularization_strength": {
                "type": "int",
                "label": "L2 shrinkage regularization strength",
                "default": 0,
            },
            "beta": {
                "type": "float",
                "label": "Beta",
                "default": 0,
            },
        },
    },
    "architectures": {
        "Basic": {
            "outputs": {
                "label": "Параметры outputs слоев",
                "collapsable": True,
                "collapsed": False,
                "layers": [],
            },
            "checkpoint": {
                "label": "Чекпоинты",
                "collapsable": True,
                "collapsed": False,
                "data": {
                    "layer": {
                        "type": "str",
                        "label": "Монитор",
                        "list": True,
                        "available": [],
                    },
                    "indicator": {
                        "type": "str",
                        "label": "Показатель",
                        "list": True,
                        "default": "Val",
                        "available": ["Train", "Val"],
                    },
                    "type": {
                        "type": "str",
                        "label": "Тип",
                        "list": True,
                        "default": "Metrics",
                        "available": [("Metrics", "Метрика"), ("Loss", "Loss")],
                    },
                    "mode": {
                        "type": "str",
                        "label": "Режим",
                        "list": True,
                        "default": "Max",
                        "available": ["Min", "Max"],
                    },
                    "save_best": {
                        "type": "bool",
                        "label": "Сохранить лучшее",
                        "default": True,
                    },
                    "save_weights": {
                        "type": "bool",
                        "label": "Сохранить веса",
                        "default": False,
                    },
                },
            },
            "callbacks": {
                "label": "Выводить",
                "collapsable": True,
                "collapsed": False,
                "layers": [],
            },
        },
        "Yolo": {},
    },
    "form": {
        "architecture": {
            "type": "str",
            "label": "Архитектура",
            "default": ArchitectureChoice.Basic,
            "list": True,
            "available": ArchitectureChoice.options(),
            "available_names": ArchitectureChoice.names(),
        },
        "optimizer": {
            "type": "str",
            "label": "Оптимизатор",
            "default": OptimizerChoice.Adam,
            "list": True,
            "available": OptimizerChoice.options(),
            "available_names": OptimizerChoice.names(),
        },
        "batch": {
            "type": "int",
            "label": "Размер батча",
            "default": 32,
        },
        "epochs": {
            "type": "int",
            "label": "Количество эпох",
            "default": 20,
        },
        "learning_rate": {
            "type": "float",
            "label": "Learning rate",
            "default": 0.0001,
        },
    },
    "callbacks": {
        "Classification": {
            "show_every_epoch": ShowEveryEpochCallback,
            "plot_loss_metric": PlotLossMetricCallback,
            "plot_metric": PlotMetricCallback,
            "plot_loss_for_classes": PlotLossForClassesCallback,
            "plot_metric_for_classes": PlotMetricForClassesCallback,
            "plot_final": PlotFinalCallback,
            "show_images": ShowImagesCallback,
        },
        "Segmentation": {
            "show_every_epoch": ShowEveryEpochCallback,
            "plot_loss_metric": PlotLossMetricCallback,
            "plot_metric": PlotMetricCallback,
            "plot_loss_for_classes": PlotLossForClassesCallback,
            "plot_metric_for_classes": PlotMetricForClassesCallback,
            "plot_final": PlotFinalCallback,
            "show_images": ShowImagesCallback,
        },
        "Regression": {
            "show_every_epoch": ShowEveryEpochCallback,
            "plot_loss_metric": PlotLossMetricCallback,
            "plot_metric": PlotMetricCallback,
            "plot_scatter": PlotScatterCallback,
            "plot_final": PlotFinalCallback,
        },
        "Timeseries": {
            "show_every_epoch": ShowEveryEpochCallback,
            "plot_loss_metric": PlotLossMetricCallback,
            "plot_metric": PlotMetricCallback,
            "plot_autocorrelation": PlotAutocorrelationCallback,
            "plot_pred_and_true": PlotPredAndTrueCallback,
            "plot_final": PlotFinalCallback,
        },
    },
    "losses": LossesData,
    "metrics": MetricsData,
}
