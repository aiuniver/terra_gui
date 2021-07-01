"""
Структура данных для предустановок обучения
"""

from typing import List

from ...mixins import AliasMixinData, UniqueListMixin


class TaskGroupData(AliasMixinData):
    losses: List
    metrics: List


class TasksGroupsList(UniqueListMixin):
    """
    Список задач и их лоссов и метрик , основанных на `TaskGroupData`
    """

    class Meta:
        source = TaskGroupData
        identifier = "alias"
