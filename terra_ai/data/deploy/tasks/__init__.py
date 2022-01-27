from typing import Any
from pydantic import validator, DirectoryPath
from pydantic.errors import EnumMemberError

from terra_ai.data.mixins import BaseMixinData

from terra_ai.data.deploy.extra import DeployTypeChoice, DeployTypePageChoice
from terra_ai.data.deploy.tasks import types


class DeployPageData(BaseMixinData):
    type: DeployTypePageChoice
    name: str


class DeployData(BaseMixinData):
    page: DeployPageData
    path_deploy: DirectoryPath
    type: DeployTypeChoice
    data: Any = {}

    @validator("type", pre=True)
    def _validate_type(cls, value: DeployTypeChoice, values) -> DeployTypeChoice:
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
        if not value:
            value = {}
        if not value.get("data"):
            value["data"] = []
        value.update({"path_deploy": values.get("path_deploy")})
        return field.type_(**value)

    @property
    def presets(self) -> dict:
        data = self.native()
        data.update({"data": self.data.presets})
        return data

    @property
    def config(self) -> dict:
        data = self.native()
        data.pop("data")
        return data

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"path_deploy"}})
        return super().dict(**kwargs)
