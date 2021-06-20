from datetime import datetime
from typing import Optional

from . import mixins, extra


class DatasetTagsData(mixins.AliasMixinData):
    name: str


class DatasetTagsListData(mixins.UniqueListMixinData):
    class Meta:
        source = DatasetTagsData
        identifier = "alias"


class DatasetData(mixins.AliasMixinData):
    name: str
    size: Optional[extra.SizeData]
    date: Optional[datetime]
    tags: DatasetTagsListData = DatasetTagsListData()


class DatasetsList(mixins.UniqueListMixinData):
    class Meta:
        source = DatasetData
        identifier = "alias"


class Project(mixins.BaseMixinData):
    datasets: DatasetsList = DatasetsList()
