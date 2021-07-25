from typing import List

from terra_ai.data.mixins import BaseMixinData

from .base import Field


class DefaultsModelingData(BaseMixinData):
    layer_form: List[Field]
    layers_types: dict


class DefaultsData(BaseMixinData):
    modeling: DefaultsModelingData
