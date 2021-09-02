from typing import List, Optional

from terra_ai.data.mixins import BaseMixinData, UniqueListMixin, AliasMixinData

from .base import Field


class DefaultsDatasetsCreationData(BaseMixinData):
    input: List[Field]
    output: List[Field]


class DefaultsDatasetsData(BaseMixinData):
    creation: DefaultsDatasetsCreationData


class DefaultsModelingData(BaseMixinData):
    layer_form: List[Field]
    layers_types: dict


class DefaultsTrainingBaseGroupData(AliasMixinData):
    name: Optional[str]
    collapsable: bool = False
    collapsed: bool = False
    fields: List[Field]


class DefaultsTrainingBaseList(UniqueListMixin):
    class Meta:
        source = DefaultsTrainingBaseGroupData
        identifier = "alias"


class DefaultsTrainingData(BaseMixinData):
    base: DefaultsTrainingBaseList


class DefaultsData(BaseMixinData):
    datasets: DefaultsDatasetsData
    modeling: DefaultsModelingData
    training: DefaultsTrainingData
