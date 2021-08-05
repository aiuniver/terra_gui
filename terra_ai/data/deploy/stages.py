from typing import Optional
from pydantic import validator
from pydantic.types import constr, PositiveInt
from transliterate import slugify

from ..mixins import BaseMixinData
from ..types import confilepath
from .extra import TaskTypeChoice


class StageUploadUserData(BaseMixinData):
    login: constr(regex=r"^[a-z]+[a-z0-9\-_]*$")
    name: str
    lastname: str
    sec: Optional[str]


class StageUploadProjectData(BaseMixinData):
    name: str
    slug: Optional[constr(regex=r"^[a-z]+[a-z0-9\-_]*$")]

    @validator("slug", always=True)
    def _validate_project_name_lat(cls, value: str, values) -> str:
        name = values.get("name")
        if not name:
            return value
        return slugify(name, language_code="ru")


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
    user: StageUploadUserData
    project: StageUploadProjectData
    task: TaskTypeChoice
    replace: bool = False
    file: StageUploadFileData
