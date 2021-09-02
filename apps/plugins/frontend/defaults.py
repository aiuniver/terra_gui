from typing import List

from terra_ai.data.mixins import BaseMixinData

from .base import Field


class DefaultsDatasetsCreationData(BaseMixinData):
    input: List[Field]
    output: List[Field]


class DefaultsDatasetsData(BaseMixinData):
    creation: DefaultsDatasetsCreationData


class DefaultsModelingData(BaseMixinData):
    layer_form: List[Field]
    layers_types: dict


class DefaultsTrainingParametersData(BaseMixinData):
    pass


class DefaultsTrainingData(BaseMixinData):
    parameters: DefaultsTrainingParametersData


class DefaultsData(BaseMixinData):
    datasets: DefaultsDatasetsData
    modeling: DefaultsModelingData
    training: DefaultsTrainingData
