"""
## Структура данных датасетов
"""

from datetime import datetime
from typing import Optional

from ..mixins import AliasMixinData, UniqueListMixin
from ..extra import SizeData
from .tags import TagsListData


class DatasetData(AliasMixinData):
    """
    Информация о датасете
    """

    name: str
    "Название"
    size: Optional[SizeData]
    "Вес"
    date: Optional[datetime]
    "Дата создания"
    tags: Optional[TagsListData] = TagsListData()
    "Список тегов"


class DatasetsList(UniqueListMixin):
    """
    Список датасетов, основанных на `DatasetData`
    ```
    class Meta:
        source = DatasetData
        identifier = "alias"
    ```
    """

    class Meta:
        source = DatasetData
        identifier = "alias"
