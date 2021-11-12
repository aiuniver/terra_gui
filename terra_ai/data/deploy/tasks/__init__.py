from typing import Any
from pathlib import Path
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
    path: DirectoryPath
    path_model: DirectoryPath
    type: DeployTypeChoice
    data: Any = {}

    def __init__(self, **data):
        page_name = data.get("page", {}).get("name", "")
        if page_name and data.get("path_model"):
            data["path_model"] = str(Path(data.get("path_model"), page_name).absolute())
        super().__init__(**data)

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
        value.update(
            {
                "path": values.get("path"),
                "path_model": values.get("path_model"),
            }
        )
        return field.type_(**value)

    @property
    def presets(self) -> dict:
        data = self.native()
        data.update({"data": self.data.presets})
        return data

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"path", "path_model"}})
        return super().dict(**kwargs)
