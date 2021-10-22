import re

from typing import Optional
from pydantic import validator
from pydantic.types import constr, PositiveInt
from pydantic.networks import HttpUrl
from transliterate import slugify

from ..mixins import BaseMixinData
from ..types import confilepath
from .extra import DeployTypeChoice, EnvVersionChoice


class StageUploadUserData(BaseMixinData):
    login: constr(regex=r"^[a-z]+[a-z0-9_]*$")
    name: str
    lastname: str
    sec: Optional[str]


class StageUploadProjectData(BaseMixinData):
    name: str
    slug: Optional[constr(regex=r"^[a-z]+[a-z0-9_]*$")]

    @validator("slug", always=True)
    def _validate_slug(cls, value: str, values) -> str:
        name = values.get("name")
        if not name:
            return value
        return re.sub(r"([\-]+)", "_", slugify(name, language_code="ru"))


class StageUploadFileData(BaseMixinData):
    path: confilepath(ext="zip")
    name: Optional[str]
    size: Optional[PositiveInt]

    @validator("name", always=True)
    def _validate_name(cls, value: str, values) -> str:
        filepath = values.get("path")
        if not filepath:
            return value
        return filepath.name

    @validator("size", always=True)
    def _validate_size(cls, value: str, values) -> str:
        filepath = values.get("path")
        if not filepath:
            return value
        return filepath.stat().st_size

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"path"}})
        return super().dict(**kwargs)


class StageUploadData(BaseMixinData):
    stage: PositiveInt
    deploy: constr(regex=r"^[a-z]+[a-z0-9\-_]*$")
    env: EnvVersionChoice = EnvVersionChoice.v1
    user: StageUploadUserData
    project: StageUploadProjectData
    task: str
    replace: bool = False
    file: StageUploadFileData


class StageCompleteData(BaseMixinData):
    stage: PositiveInt
    deploy: constr(regex=r"^[a-z]+[a-z0-9\-_]*$")
    login: constr(regex=r"^[a-z]+[a-z0-9\-_]*$")
    project: constr(regex=r"^[a-z]+[a-z0-9\-_]*$")


class StageResponseData(BaseMixinData):
    stage: PositiveInt
    success: bool
    url: HttpUrl
    api_text: str
