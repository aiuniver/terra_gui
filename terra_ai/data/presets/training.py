"""
Предустановки обучения
"""

from ..presets.extra.training import TasksGroupsList
from ..training.extra import TaskChoice, LossChoice, MetricChoice


TasksGroups = TasksGroupsList(
    [
        {
            "alias": TaskChoice.classification.value,
            "losses": [
                LossChoice.categorical_crossentropy.value,
                LossChoice.binary_crossentropy.value,
                LossChoice.mse.value,
                LossChoice.squared_hinge.value,
                LossChoice.hinge.value,
                LossChoice.categorical_hinge.value,
                LossChoice.sparse_categorical_crossentropy.value,
                LossChoice.kl_divergence.value,
                LossChoice.poisson.value,
            ],
            "metrics": [
                MetricChoice.accuracy.value,
                MetricChoice.binary_accuracy.value,
                MetricChoice.binary_crossentropy.value,
                MetricChoice.categorical_accuracy.value,
                MetricChoice.categorical_crossentropy.value,
                MetricChoice.sparse_categorical_accuracy.value,
                MetricChoice.sparse_categorical_crossentropy.value,
                MetricChoice.top_k_categorical_accuracy.value,
                MetricChoice.sparse_top_k_categorical_accuracy.value,
                MetricChoice.hinge.value,
                MetricChoice.kullback_leibler_divergence.value,
                MetricChoice.poisson.value,
            ],
        },
        {
            "alias": TaskChoice.segmentation.value,
            "losses": [
                LossChoice.categorical_crossentropy.value,
                LossChoice.binary_crossentropy.value,
                LossChoice.squared_hinge.value,
                LossChoice.hinge.value,
                LossChoice.categorical_hinge.value,
                LossChoice.sparse_categorical_crossentropy.value,
                LossChoice.kl_divergence.value,
                LossChoice.poisson.value,
            ],
            "metrics": [
                MetricChoice.dice_coef.value,
                MetricChoice.mean_io_u.value,
                MetricChoice.accuracy.value,
                MetricChoice.binary_accuracy.value,
                MetricChoice.binary_crossentropy.value,
                MetricChoice.categorical_accuracy.value,
                MetricChoice.categorical_crossentropy.value,
                MetricChoice.sparse_categorical_accuracy.value,
                MetricChoice.sparse_categorical_crossentropy.value,
                MetricChoice.top_k_categorical_accuracy.value,
                MetricChoice.sparse_top_k_categorical_accuracy.value,
                MetricChoice.hinge.value,
                MetricChoice.kullback_leibler_divergence.value,
                MetricChoice.poisson.value,
            ],
        },
        {
            "alias": TaskChoice.regression.value,
            "losses": [
                LossChoice.mse.value,
                LossChoice.mae.value,
                LossChoice.mape.value,
                LossChoice.msle.value,
                LossChoice.log_cosh.value,
                LossChoice.cosine_similarity.value,
            ],
            "metrics": [
                MetricChoice.accuracy.value,
                MetricChoice.mae.value,
                MetricChoice.mse.value,
                MetricChoice.mape.value,
                MetricChoice.msle.value,
                MetricChoice.logcosh.value,
                MetricChoice.cosine_similarity.value,
            ],
        },
        {
            "alias": TaskChoice.timeseries.value,
            "losses": [
                LossChoice.mse.value,
                LossChoice.mae.value,
                LossChoice.mape.value,
                LossChoice.msle.value,
                LossChoice.log_cosh.value,
                LossChoice.cosine_similarity.value,
            ],
            "metrics": [
                MetricChoice.accuracy.value,
                MetricChoice.mae.value,
                MetricChoice.mse.value,
                MetricChoice.mape.value,
                MetricChoice.msle.value,
                MetricChoice.logcosh.value,
                MetricChoice.cosine_similarity.value,
            ],
        },
    ]
)
