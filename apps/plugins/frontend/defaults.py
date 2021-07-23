from terra_ai.data.mixins import BaseMixinData


class DefaultsModelingData(BaseMixinData):
    layers_types: dict


class DefaultsData(BaseMixinData):
    modeling: DefaultsModelingData
