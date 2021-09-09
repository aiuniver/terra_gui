from typing import Optional
from pydantic import validator

from ..mixins import BaseMixinData, UniqueListMixin
from ..types import confilepath
from ... import settings


class ProjectListData(BaseMixinData):
    """
    Информация о проекте в списке
    """

    value: confilepath(ext=settings.PROJECT_EXT)
    "Путь к файлу проекта"
    label: Optional[str]
    "Название"

    @validator("label", allow_reuse=True, always=True)
    def _validate_label(cls, value: str, values) -> str:
        file_path = values.get("value")
        if not file_path:
            return value
        return file_path.name.split(f".{settings.PROJECT_EXT}")[0]


class ProjectsList(UniqueListMixin):
    """
    Список проектов, основанных на `ProjectListData`
    ```
    class Meta:
        source = ProjectListData
        identifier = "label"
    ```
    """

    class Meta:
        source = ProjectListData
        identifier = "label"


class ProjectsInfoData(BaseMixinData):
    """
    Информация о проектах
    """

    projects: ProjectsList = ProjectsList()
