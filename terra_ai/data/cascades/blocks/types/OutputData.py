from typing import Optional
from pydantic import validator
from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerInputTypeChoice


class ParametersMainData(BaseMixinData):
    type: LayerInputTypeChoice = LayerInputTypeChoice.Image
    width: Optional[PositiveInt] = 640
    height: Optional[PositiveInt] = 480

    def __init__(self, **data):
        _type = data.get("type")
        _keys = ["type"]
        if _type == LayerInputTypeChoice.Video:
            _keys += ["width", "height"]
        data = dict(filter(lambda item: item[0] in _keys, data.items()))
        super().__init__(**data)

    @validator("type", pre=True)
    def _validate_type(cls, value: LayerInputTypeChoice) -> LayerInputTypeChoice:
        for name, item in cls.__fields__.items():
            if name in ["type"]:
                continue
            cls.__fields__[name].required = False
        if value == LayerInputTypeChoice.Video:
            cls.__fields__["width"].required = True
            cls.__fields__["height"].required = True
        return value
