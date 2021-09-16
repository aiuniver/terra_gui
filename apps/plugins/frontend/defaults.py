from typing import List, Dict, Optional, Union

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


class DefaultsTrainingBaseGroupData(BaseMixinData):
    name: Optional[str]
    collapsable: bool = False
    collapsed: bool = False
    fields: Union[List[Field], Dict[str, List[Field]]]


class DefaultsTrainingBaseData(BaseMixinData):
    main: DefaultsTrainingBaseGroupData
    fit: DefaultsTrainingBaseGroupData
    optimizer: DefaultsTrainingBaseGroupData
    outputs: DefaultsTrainingBaseGroupData
    checkpoint: DefaultsTrainingBaseGroupData


class DefaultsTrainingData(BaseMixinData):
    base: DefaultsTrainingBaseData


class DefaultsData(BaseMixinData):
    datasets: DefaultsDatasetsData
    modeling: DefaultsModelingData
    training: DefaultsTrainingData
