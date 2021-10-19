from datetime import datetime, date
from typing import List, Optional
from pydantic import validator, DirectoryPath
# from pathlib import Path
from enum import Enum
# import os
from terra_ai.data.mixins import AliasMixinData
from terra_ai.data.extra import FileSizeData
# from terra_ai.data.mixins import BaseMixinData


ANNOT_EXT = 'annot'


# class AnnotationPathsData(BaseMixinData):
#     basepath: DirectoryPath
    # sources: Optional[DirectoryPath]
    #
    # @validator(
    #     "sources",
    #     always=True,
    # )
    # def _validate_internal_path(cls, value, values, field) -> Path:
    #     path = Path(values.get("basepath"), field.name)
    #     os.makedirs(path, exist_ok=True)
    #     return path


class MarkupTypeChoice(str, Enum):
    classification = "classification"
    tracker = "tracker"


class AnnotationData(AliasMixinData):
    """
    Информация о разметке
    """

    alias: str
    created: Optional[datetime]
    until: Optional[date]
    classes_names: List
    classes_colors: List
    task_type: MarkupTypeChoice
    progress: List[int]
    size: Optional[FileSizeData]
    cover: str
    to_do: List[str]


class MarkupData(AliasMixinData):
    """
    Входная информация о разметке данных
    """

    alias: str
    "Название проекта"
    annotations_path: DirectoryPath
    "Путь к директории"
    source: str
    "Название архива"
    created: Optional[datetime]
    "Время создания проекта"
    until: List[int]
    "Срок сдачи проекта"
    classes_names: List
    "Имена классов"
    classes_colors: List
    "Цвета"
    task_type: MarkupTypeChoice
    "Тип задачи"
    to_do: List[str]
    "Ответственные"
