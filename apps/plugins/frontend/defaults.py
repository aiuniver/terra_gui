from typing import List, Dict

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice

from .base import Field


class DefaultsDatasetsCreationLayersData(BaseMixinData):
    base: List[Field]


class DefaultsDatasetsCreationInputData(DefaultsDatasetsCreationLayersData):
    type: Dict[LayerInputTypeChoice, List[Field]]


class DefaultsDatasetsCreationOutputData(DefaultsDatasetsCreationLayersData):
    type: Dict[LayerOutputTypeChoice, List[Field]]


class DefaultsDatasetsCreationData(BaseMixinData):
    inputs: DefaultsDatasetsCreationInputData
    outputs: DefaultsDatasetsCreationOutputData


class DefaultsDatasetsData(BaseMixinData):
    creation: DefaultsDatasetsCreationData


class DefaultsModelingData(BaseMixinData):
    layer_form: List[Field]
    layers_types: dict


class DefaultsData(BaseMixinData):
    datasets: DefaultsDatasetsData
    modeling: DefaultsModelingData
