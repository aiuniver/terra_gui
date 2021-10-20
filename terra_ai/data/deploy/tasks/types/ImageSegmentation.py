from pydantic import FilePath

from terra_ai.data.mixins import BaseMixinData, UniqueListMixin


class Data(BaseMixinData):
    source: FilePath


class DataList(UniqueListMixin):
    class Meta:
        source = Data
        identifier = "source"
