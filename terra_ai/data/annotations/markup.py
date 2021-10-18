from datetime import datetime
from typing import List, Optional
from enum import Enum
from terra_ai.data.mixins import AliasMixinData
from terra_ai.data.extra import FileSizeData

ANNOT_EXT = 'annot'
SOURCE = '/content/drive/MyDrive/TerraAI/annotations/sources'
DESTINATION_PATH = '/content/drive/MyDrive/TerraAI/datasets/sources'
ANNOT_DIRECTORY = '/content/drive/MyDrive/TerraAI/annotations'


class MarkupTypeChoice(str, Enum):
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
    task_type: MarkupTypeChoice
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
    task_type: MarkupTypeChoice
    "Тип задачи"
    to_do: List[str]
    "Ответственные"
