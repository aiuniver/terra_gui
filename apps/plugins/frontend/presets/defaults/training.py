from terra_ai.data.training.extra import (
    OptimizerChoice,
    CheckpointIndicatorChoice,
    CheckpointTypeChoice,
    ArchitectureChoice,
)

from ...extra import FieldTypeChoice


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
}


ArchitectureOptimizerExtraFields = {
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
}


ArchitectureGroupMain = {
    "fields": [
        {
            "type": "auto_complete",
            "label": "Оптимизатор",
            "name": "optimizer",
            "parse": "optimizer[type]",
            "changeable": True,
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
            "name": "optimizer_main_learning_rate",
            "parse": "optimizer[parameters][main][learning_rate]",
            "value": 0.001,
        },
    ],
}


ArchitectureGroupGANFit = {
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
            "name": "optimizer_main_learning_rate",
            "parse": "optimizer[parameters][main][learning_rate]",
            "value": 0.0002,
        },
    ],
}


ArchitectureGroupAutobalanceFit = {
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
            "name": "optimizer_main_learning_rate",
            "parse": "optimizer[parameters][main][learning_rate]",
            "value": 0.001,
        },
        {
            "type": "checkbox",
            "label": "Автобалансировка",
            "name": "autobalance",
            "parse": "[autobalance]",
            "value": False,
        },
    ],
}


ArchitectureGroupFitYolo = {
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
            "name": "optimizer_main_learning_rate",
            "parse": "optimizer[parameters][main][learning_rate]",
            "value": 0.001,
            "visible": False,
        },
    ],
}


ArchitectureGroupOptimizer = {
    "name": "Параметры оптимизатора",
    "collapsable": True,
    "collapsed": True,
    "fields": [],
}


ArchitectureGroupYoloField = [
    {
        "type": "number",
        "label": "Learning rate init",
        "name": "architecture_parameters_yolo_train_lr_init",
        "parse": "architecture[parameters][yolo][train_lr_init]",
        "value": 1e-4,
    },
    {
        "type": "number",
        "label": "Learning rate end",
        "name": "architecture_parameters_yolo_train_lr_end",
        "parse": "architecture[parameters][yolo][train_lr_end]",
        "value": 1e-6,
    },
    {
        "type": "number",
        "label": "Yolo IoU loss threshold",
        "name": "architecture_parameters_yolo_yolo_iou_loss_thresh",
        "parse": "architecture[parameters][yolo][yolo_iou_loss_thresh]",
        "value": 0.5,
    },
    {
        "type": "number",
        "label": "Warmup epochs",
        "name": "architecture_parameters_yolo_train_warmup_epochs",
        "parse": "architecture[parameters][yolo][train_warmup_epochs]",
        "value": 2,
    },
]


ArchitectureGroupYoloV3 = {
    "name": "Параметры YoloV3",
    "collapsable": True,
    "collapsed": False,
    "fields": ArchitectureGroupYoloField,
}


ArchitectureGroupYoloV4 = {
    "name": "Параметры YoloV4",
    "collapsable": True,
    "collapsed": False,
    "fields": ArchitectureGroupYoloField,
}


ArchitectureBasicForm = {
    "main": ArchitectureGroupMain,
    "fit": ArchitectureGroupFit,
    "optimizer": ArchitectureGroupOptimizer,
    "outputs": {
        "name": "Параметры выходных слоев",
        "collapsable": True,
        "collapsed": False,
        "fields": {},
    },
    "checkpoint": {
        "name": "Чекпоинты",
        "collapsable": True,
        "collapsed": False,
        "fields": [
            {
                "type": "select",
                "label": "Функция",
                "name": "architecture_parameters_checkpoint_metric_name",
                "parse": "architecture[parameters][checkpoint][metric_name]",
                "list": [],
            },
            {
                "type": "select",
                "label": "Монитор",
                "name": "architecture_parameters_checkpoint_layer",
                "parse": "architecture[parameters][checkpoint][layer]",
                "list": [],
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
        ],
    },
}


ArchitectureBasicAutobalanceForm = {
    "main": ArchitectureGroupMain,
    "fit": ArchitectureGroupAutobalanceFit,
    "optimizer": ArchitectureGroupOptimizer,
    "outputs": {
        "name": "Параметры выходных слоев",
        "collapsable": True,
        "collapsed": False,
        "fields": {},
    },
    "checkpoint": {
        "name": "Чекпоинты",
        "collapsable": True,
        "collapsed": False,
        "fields": [
            {
                "type": "select",
                "label": "Функция",
                "name": "architecture_parameters_checkpoint_metric_name",
                "parse": "architecture[parameters][checkpoint][metric_name]",
                "list": [],
            },
            {
                "type": "select",
                "label": "Монитор",
                "name": "architecture_parameters_checkpoint_layer",
                "parse": "architecture[parameters][checkpoint][layer]",
                "list": [],
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
        ],
    },
}


ArchitectureYoloForm = {
    "main": ArchitectureGroupMain,
    "fit": ArchitectureGroupFitYolo,
    "optimizer": ArchitectureGroupOptimizer,
    "outputs": {
        "name": "Параметры выходных слоев",
        "collapsable": True,
        "collapsed": False,
        "visible": False,
        "fields": [],
    },
    "checkpoint": {
        "name": "Чекпоинты",
        "collapsable": True,
        "collapsed": False,
        "visible": False,
        "fields": [
            {
                "type": "select",
                "label": "Функция",
                "name": "architecture_parameters_checkpoint_metric_name",
                "parse": "architecture[parameters][checkpoint][metric_name]",
                "list": [],
            },
            {
                "type": "select",
                "label": "Монитор",
                "name": "architecture_parameters_checkpoint_layer",
                "parse": "architecture[parameters][checkpoint][layer]",
                "list": [],
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
        ],
    },
}


ArchitectureGANForm = {
    "main": ArchitectureGroupMain,
    "fit": ArchitectureGroupGANFit,
    "optimizer": ArchitectureGroupOptimizer,
    "outputs": {
        "name": "Параметры выходных слоев",
        "collapsable": True,
        "collapsed": False,
        "fields": {},
    },
    "checkpoint": {
        "name": "Чекпоинты",
        "collapsable": True,
        "collapsed": False,
        "fields": [
            {
                "type": "number",
                "label": "Интервал эпох для сохранения",
                "name": "architecture_parameters_checkpoint_epoch_interval",
                "parse": "architecture[parameters][checkpoint][epoch_interval]",
                "value": 10,
            }
        ],
    },
}


Architectures = {
    ArchitectureChoice.Basic: {**ArchitectureBasicForm},
    ArchitectureChoice.ImageClassification: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.ImageSegmentation: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.TextClassification: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.TextSegmentation: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.DataframeClassification: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.DataframeRegression: {**ArchitectureBasicForm},
    ArchitectureChoice.Timeseries: {**ArchitectureBasicForm},
    ArchitectureChoice.TimeseriesTrend: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.AudioClassification: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.VideoClassification: {**ArchitectureBasicAutobalanceForm},
    ArchitectureChoice.YoloV3: {
        **ArchitectureYoloForm,
        "yolo": ArchitectureGroupYoloV3,
    },
    ArchitectureChoice.YoloV4: {
        **ArchitectureYoloForm,
        "yolo": ArchitectureGroupYoloV4,
    },
    ArchitectureChoice.Tracker: {**ArchitectureBasicForm},
    ArchitectureChoice.ImageGAN: {**ArchitectureGANForm},
    ArchitectureChoice.ImageCGAN: {**ArchitectureGANForm},
    ArchitectureChoice.TextToImageGAN: {**ArchitectureGANForm},
    ArchitectureChoice.ImageToImageGAN: {**ArchitectureGANForm},
    ArchitectureChoice.ImageSRGAN: {**ArchitectureGANForm},
}
