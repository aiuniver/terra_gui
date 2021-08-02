from typing import Optional
from pydantic import validator
from pydantic.types import PositiveInt
from transliterate import slugify

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import confilepath


class StageUploadUserData(BaseMixinData):
    login: str
    name: str
    lastname: str


class StageUploadFileData(BaseMixinData):
    path: confilepath(ext="zip")
    name: Optional[str]
    size: Optional[PositiveInt]

    @validator("name", always=True)
    def _validate_name(cls, value: str, values) -> str:
        filepath = values.get("path")
        return filepath.name

    @validator("size", always=True)
    def _validate_size(cls, value: str, values) -> str:
        filepath = values.get("path")
        return filepath.stat().st_size

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"path"}})
        return super().dict(**kwargs)


class StageUploadData(BaseMixinData):
    stage: PositiveInt
    user: StageUploadUserData
    project_name: str
    project_name_lat: Optional[str]
    url: str
    replace: bool = False
    file: StageUploadFileData

    @validator("project_name_lat", always=True)
    def _validate_project_name_lat(cls, value: str, values) -> str:
        project_name = values.get("project_name")
        if not project_name:
            return value
        return slugify(project_name, language_code="ru")
