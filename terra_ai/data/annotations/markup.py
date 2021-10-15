from datetime import datetime
from typing import List, Optional
from enum import Enum
from terra_ai.data.mixins import AliasMixinData
from terra_ai.data.extra import FileSizeData


class MarkupChoice(str, Enum):
    classification = "classification"
    tracker = "tracker"


class AnnotationData(AliasMixinData):
    """
    Информация о разметке
    """

    alias: str
    created: Optional[datetime]
    until: str
    classes_names: List
    classes_colors: List
    task_type: MarkupChoice
    progress: List[int]
    size: Optional[FileSizeData]
    cover: str
    to_do: List[str]


class MarkupData(AliasMixinData):
    """
    Полная информация о разметке данных
    """

    alias: str
    "Название проекта"
    source: str
    "Путь к директории разметки проекта"
    created: Optional[datetime]
    "Время создания проекта"
    until: str
    "Срок сдачи проекта"
    classes_names: List
    "Имена классов"
    classes_colors: List
    "Цвета"
    task_type: MarkupChoice
    "Тип задачи"
    to_do: List[str]
    "Ответственные"
