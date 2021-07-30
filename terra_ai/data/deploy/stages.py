from typing import Optional
from pydantic import validator
from pydantic.types import PositiveInt
from transliterate import slugify

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import confilename


class StageUploadUserData(BaseMixinData):
    login: str
    name: str
    lastname: str


class StageUploadData(BaseMixinData):
    stage: PositiveInt
    user: StageUploadUserData
    project_name: str
    project_name_lat: Optional[str]
    url: str
    replace: bool = False
    filename: confilename(ext="zip")
    filesize: PositiveInt

    @validator("project_name_lat", always=True)
    def _validate_project_name_lat(cls, value: str, values) -> str:
        project_name = values.get("project_name")
        if not project_name:
            return value
        value = slugify(project_name, language_code="ru")
        return value
