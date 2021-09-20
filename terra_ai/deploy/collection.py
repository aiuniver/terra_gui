from typing import Optional, List

from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.deploy import tasks
from terra_ai.data.deploy.extra import TaskTypeChoice
from terra_ai.data.deploy.tasks import BaseCollection
from terra_ai.data.deploy.collection import CollectionData


TASKS_RELATIONS = {
    TaskTypeChoice.image_classification: {"ImageClassification"},
}


class Collection:
    __dataset: Optional[DatasetData] = None
    __type: Optional[TaskTypeChoice] = None
    __data: List[BaseCollection] = []

    @property
    def data(self) -> Optional[CollectionData]:
        if not self.__type:
            return
        data = CollectionData(type=self.__type)
        data.data = self.__data
        return data

    def __define(self):
        __model = self.__dataset.model
        __tasks = []
        for __input in __model.inputs:
            for __output in __model.outputs:
                __tasks.append(f"{__input.task.name}{__output.task.name}")
        try:
            self.__type = list(TASKS_RELATIONS.keys())[
                list(TASKS_RELATIONS.values()).index(set(__tasks))
            ]
        except Exception:
            return

        for __task in __tasks:
            _task_class = getattr(tasks, f"{__task}Collection", None)
            if not _task_class:
                continue
            task_instance = _task_class(dataset=self.__dataset)
            self.__data.append(task_instance)

    def __clear(self):
        self.__dataset = None
        self.__type = None
        self.__data = []

    def update(self, dataset: Optional[DatasetData] = None):
        self.__clear()
        if not dataset:
            return
        self.__dataset = dataset
        self.__define()
