from datetime import datetime
from typing import Optional

from . import mixins


class DatasetTagsData(mixins.AliasMixinData):
    name: str


class DatasetTagsListData(mixins.ListMixinData):
    class Meta:
        source = DatasetTagsData
        identifier = "alias"


class DatasetData(mixins.AliasMixinData, mixins.ListOfDictMixinData):
    class Meta:
        lists_of_dict = ["tags"]

    name: str
    size: Optional[mixins.SizeData]
    date: Optional[datetime]
    tags: DatasetTagsListData = DatasetTagsListData()


class DatasetsList(mixins.ListMixinData):
    class Meta:
        source = DatasetData
        identifier = "alias"


class Project(mixins.ListOfDictMixinData):
    class Meta:
        lists_of_dict = ["datasets"]

    datasets: DatasetsList = DatasetsList()
