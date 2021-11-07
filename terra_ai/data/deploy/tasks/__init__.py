from typing import Any
from pydantic import validator, DirectoryPath
from pydantic.errors import EnumMemberError

from terra_ai.data.mixins import BaseMixinData

from terra_ai.data.deploy.extra import DeployTypeChoice
from terra_ai.data.deploy.tasks import types


class DeployData(BaseMixinData):
    path: DirectoryPath
    type: DeployTypeChoice
    data: Any = {}

    @validator("type", pre=True)
    def _validate_type(cls, value: DeployTypeChoice) -> DeployTypeChoice:
        if value not in list(DeployTypeChoice):
            raise EnumMemberError(enum_values=list(DeployTypeChoice))
        name = (
            value if isinstance(value, DeployTypeChoice) else DeployTypeChoice(value)
        ).name
        type_ = getattr(types, name).Data
        cls.__fields__["data"].type_ = type_
        return value

    @validator("data", always=True)
    def _validate_data(cls, value: Any, values, field) -> Any:
        value.update({"path": values.get("path")})
        return field.type_(**value)

    @property
    def presets(self) -> dict:
        data = self.native()
        data.update({"data": self.data.presets})
        return data

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"path"}})
        return super().dict(**kwargs)
