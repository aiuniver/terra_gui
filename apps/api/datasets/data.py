from apps.plugins.terra.frontend.mixins import BaseMixinData, UniqueListMixin
from apps.plugins.terra.frontend.types import confilepath


class DatasetSourceData(BaseMixinData):
    value: confilepath(ext="zip")


class DatasetsSourcesList(UniqueListMixin):
    class Meta:
        source = DatasetSourceData
        identifier = "value"

    def list(self) -> list:
        return list(map(lambda item: item.value.name, self))
