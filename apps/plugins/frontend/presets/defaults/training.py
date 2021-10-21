from terra_ai.data.training.extra import (
    OptimizerChoice,
    CheckpointIndicatorChoice,
    CheckpointTypeChoice,
    CheckpointModeChoice,
    ArchitectureChoice,
)
from terra_ai.data.training.extra import TasksGroupsList
from terra_ai.data.presets.training import TasksGroups

from ...extra import FieldTypeChoice


TrainingTasksRelations = TasksGroupsList(TasksGroups)


TrainingLossSelect = {
    "type": FieldTypeChoice.select.value,
    "label": "Loss",
    "name": "architecture_parameters_outputs_%i_loss",
    "parse": "architecture[parameters][outputs][%i][loss]",
}


TrainingMetricSelect = {
    "type": FieldTypeChoice.multiselect.value,
    "label": "Выберите метрики",
    "name": "architecture_parameters_outputs_%i_metrics",
    "parse": "architecture[parameters][outputs][%i][metrics]",
}


TrainingClassesQuantitySelect = {
    "type": FieldTypeChoice.number.value,
    "label": "Количество классов",
    "name": "architecture_parameters_outputs_%i_classes_quantity",
    "parse": "architecture[parameters][outputs][%i][classes_quantity]",
    "disabled": True,
}


ArchitectureGroupMain = {
    "fields": [
        {
            "type": "auto_complete",
            "label": "Оптимизатор",
            "name": "optimizer",
            "parse": "optimizer[type]",
            "value": OptimizerChoice.Adam.name,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    list(OptimizerChoice),
                )
            ),
        },
    ],
}


ArchitectureGroupFit = {
    "fields": [
        {
            "type": "number",
            "label": "Размер батча",
            "name": "batch",
            "parse": "[batch]",
            "value": 32,
        },
        {
            "type": "number",
            "label": "Количество эпох",
            "name": "epochs",
            "parse": "[epochs]",
            "value": 20,
        },
        {
            "type": "number",
            "label": "Learning rate",
            "name": "optimizer_learning_rate",
            "parse": "optimizer[parameters][main][learning_rate]",
            "value": 0.001,
        },
    ],
}


ArchitectureGroupOptimizer = {
    "name": "Параметры оптимизатора",
    "collapsable": True,
    "collapsed": True,
    "fields": {
        OptimizerChoice.SGD.name: [
            {
                "type": "number",
                "label": "Momentum",
                "name": "optimizer_extra_momentum",
                "parse": "optimizer[parameters][extra][momentum]",
                "value": 0,
            },
            {
                "type": "checkbox",
                "label": "Nesterov",
                "name": "optimizer_extra_nesterov",
                "parse": "optimizer[parameters][extra][nesterov]",
                "value": False,
            },
        ],
        OptimizerChoice.RMSprop.name: [
            {
                "type": "number",
                "label": "RHO",
                "name": "optimizer_extra_rho",
                "parse": "optimizer[parameters][extra][rho]",
                "value": 0.9,
            },
            {
                "type": "number",
                "label": "Momentum",
                "name": "optimizer_extra_momentum",
                "parse": "optimizer[parameters][extra][momentum]",
                "value": 0,
            },
            {
                "type": "number",
                "label": "Epsilon",
                "name": "optimizer_extra_epsilon",
                "parse": "optimizer[parameters][extra][epsilon]",
                "value": 1e-07,
            },
            {
                "type": "checkbox",
                "label": "Centered",
                "name": "optimizer_extra_centered",
                "parse": "optimizer[parameters][extra][centered]",
                "value": False,
            },
        ],
        OptimizerChoice.Adam.name: [
            {
                "type": "number",
                "label": "Beta 1",
                "name": "optimizer_extra_beta_1",
                "parse": "optimizer[parameters][extra][beta_1]",
                "value": 0.9,
            },
            {
                "type": "number",
                "label": "Beta 2",
                "name": "optimizer_extra_beta_2",
                "parse": "optimizer[parameters][extra][beta_2]",
                "value": 0.999,
            },
            {
                "type": "number",
                "label": "Epsilon",
                "name": "optimizer_extra_epsilon",
                "parse": "optimizer[parameters][extra][epsilon]",
                "value": 1e-07,
            },
            {
                "type": "checkbox",
                "label": "Amsgrad",
                "name": "optimizer_extra_amsgrad",
                "parse": "optimizer[parameters][extra][amsgrad]",
                "value": False,
            },
        ],
        OptimizerChoice.Adadelta.name: [
            {
                "type": "number",
                "label": "RHO",
                "name": "optimizer_extra_rho",
                "parse": "optimizer[parameters][extra][rho]",
                "value": 0.95,
            },
            {
                "type": "number",
                "label": "Epsilon",
                "name": "optimizer_extra_epsilon",
                "parse": "optimizer[parameters][extra][epsilon]",
                "value": 1e-07,
            },
        ],
        OptimizerChoice.Adagrad.name: [
            {
                "type": "number",
                "label": "Initial accumulator value",
                "name": "optimizer_extra_initial_accumulator_value",
                "parse": "optimizer[parameters][extra][initial_accumulator_value]",
                "value": 0.1,
            },
            {
                "type": "number",
                "label": "Epsilon",
                "name": "optimizer_extra_epsilon",
                "parse": "optimizer[parameters][extra][epsilon]",
                "value": 1e-07,
            },
        ],
        OptimizerChoice.Adamax.name: [
            {
                "type": "number",
                "label": "Beta 1",
                "name": "optimizer_extra_beta_1",
                "parse": "optimizer[parameters][extra][beta_1]",
                "value": 0.9,
            },
            {
                "type": "number",
                "label": "Beta 2",
                "name": "optimizer_extra_beta_2",
                "parse": "optimizer[parameters][extra][beta_2]",
                "value": 0.999,
            },
            {
                "type": "number",
                "label": "Epsilon",
                "name": "optimizer_extra_epsilon",
                "parse": "optimizer[parameters][extra][epsilon]",
                "value": 1e-07,
            },
        ],
        OptimizerChoice.Nadam.name: [
            {
                "type": "number",
                "label": "Beta 1",
                "name": "optimizer_extra_beta_1",
                "parse": "optimizer[parameters][extra][beta_1]",
                "value": 0.9,
            },
            {
                "type": "number",
                "label": "Beta 2",
                "name": "optimizer_extra_beta_2",
                "parse": "optimizer[parameters][extra][beta_2]",
                "value": 0.999,
            },
            {
                "type": "number",
                "label": "Epsilon",
                "name": "optimizer_extra_epsilon",
                "parse": "optimizer[parameters][extra][epsilon]",
                "value": 1e-07,
            },
        ],
        OptimizerChoice.Ftrl.name: [
            {
                "type": "number",
                "label": "Learning rate power",
                "name": "optimizer_extra_learning_rate_power",
                "parse": "optimizer[parameters][extra][learning_rate_power]",
                "value": -0.5,
            },
            {
                "type": "number",
                "label": "Initial accumulator value",
                "name": "optimizer_extra_initial_accumulator_value",
                "parse": "optimizer[parameters][extra][initial_accumulator_value]",
                "value": 0.1,
            },
            {
                "type": "number",
                "label": "L1 regularization strength",
                "name": "optimizer_extra_l1_regularization_strength",
                "parse": "optimizer[parameters][extra][l1_regularization_strength]",
                "value": 0,
            },
            {
                "type": "number",
                "label": "L2 regularization strength",
                "name": "optimizer_extra_l2_regularization_strength",
                "parse": "optimizer[parameters][extra][l2_regularization_strength]",
                "value": 0,
            },
            {
                "type": "number",
                "label": "L2 shrinkage regularization strength",
                "name": "optimizer_extra_l2_shrinkage_regularization_strength",
                "parse": "optimizer[parameters][extra][l2_shrinkage_regularization_strength]",
                "value": 0,
            },
            {
                "type": "number",
                "label": "Beta",
                "name": "optimizer_extra_beta",
                "parse": "optimizer[parameters][extra][beta]",
                "value": 0,
            },
        ],
    },
}


Architectures = {
    ArchitectureChoice.Basic: {
        "main": ArchitectureGroupMain,
        "fit": ArchitectureGroupFit,
        "optimizer": ArchitectureGroupOptimizer,
        "outputs": {
            "name": "Параметры выходных слоев",
            "collapsable": True,
            "collapsed": False,
            "fields": [],
        },
        "checkpoint": {
            "name": "Чекпоинты",
            "collapsable": True,
            "collapsed": False,
            "fields": [
                {
                    "type": "select",
                    "label": "Монитор",
                    "name": "architecture_parameters_checkpoint_layer",
                    "parse": "architecture[parameters][checkpoint][layer]",
                },
                {
                    "type": "select",
                    "label": "Indicator",
                    "name": "architecture_parameters_checkpoint_indicator",
                    "parse": "architecture[parameters][checkpoint][indicator]",
                    "value": CheckpointIndicatorChoice.Val.name,
                    "list": list(
                        map(
                            lambda item: {"value": item.name, "label": item.value},
                            list(CheckpointIndicatorChoice),
                        )
                    ),
                },
                {
                    "type": "select",
                    "label": "Тип",
                    "name": "architecture_parameters_checkpoint_type",
                    "parse": "architecture[parameters][checkpoint][type]",
                    "value": CheckpointTypeChoice.Metrics.name,
                    "list": list(
                        map(
                            lambda item: {"value": item.name, "label": item.value},
                            list(CheckpointTypeChoice),
                        )
                    ),
                },
                {
                    "type": "select",
                    "label": "Режим",
                    "name": "architecture_parameters_checkpoint_mode",
                    "parse": "architecture[parameters][checkpoint][mode]",
                    "value": CheckpointModeChoice.Max.name,
                    "list": list(
                        map(
                            lambda item: {"value": item.name, "label": item.value},
                            list(CheckpointModeChoice),
                        )
                    ),
                },
                # {
                #     "type": "checkbox",
                #     "label": "Сохранить лучшее",
                #     "name": "architecture_parameters_checkpoint_save_best",
                #     "parse": "architecture[parameters][checkpoint][save_best]",
                #     "value": True,
                # },
                # {
                #     "type": "checkbox",
                #     "label": "Сохранить веса",
                #     "name": "architecture_parameters_checkpoint_save_weights",
                #     "parse": "architecture[parameters][checkpoint][save_weights]",
                #     "value": False,
                # },
            ],
        },
    },
    ArchitectureChoice.YoloV3: {
        "main": ArchitectureGroupMain,
        "fit": ArchitectureGroupFit,
        "optimizer": ArchitectureGroupOptimizer,
    },
}
