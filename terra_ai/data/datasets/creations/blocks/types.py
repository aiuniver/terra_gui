from typing import Any, List, Optional
from pydantic import validator
from pydantic.errors import EnumMemberError

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerHandlerChoice, LayerSelectTypeChoice
from terra_ai.data.datasets.creations.blocks import handlers


class BlockDataParameters(BaseMixinData):
    type: Optional[LayerSelectTypeChoice]
    data: List[str] = []


class BlockHandlerParameters(BaseMixinData):
    type: LayerHandlerChoice
    options: Any

    @validator("type", pre=True)
    def _validate_type(cls, value: LayerHandlerChoice) -> LayerHandlerChoice:
        if value not in list(LayerHandlerChoice):
            raise EnumMemberError(enum_values=list(LayerHandlerChoice))
        name = (
            value
            if isinstance(value, LayerHandlerChoice)
            else LayerHandlerChoice(value)
        ).name
        type_ = getattr(handlers, name)
        cls.__fields__["options"].required = True
        cls.__fields__["options"].type_ = type_.OptionsData
        return value

    @validator("options", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class BlockLayerParameters(BaseMixinData):
    pass
