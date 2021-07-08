import os

from pathlib import Path
from pydantic import validator, BaseModel, DirectoryPath

from django.conf import settings


UNKNOWN_NAME = "NoName"


class ProjectPath(BaseModel):
    sources: DirectoryPath
    datasets: DirectoryPath
    modeling: DirectoryPath
    training: DirectoryPath

    @validator("sources", "datasets", "modeling", "training", pre=True)
    def _validate_path(cls, value: DirectoryPath) -> DirectoryPath:
        try:
            os.makedirs(value)
        except FileExistsError:
            pass
        return value


class Project(BaseModel):
    name: str = UNKNOWN_NAME
    path: ProjectPath = ProjectPath(
        sources=Path(settings.TERRA_AI_DATA_PATH, "datasets", "sources"),
        datasets=Path(settings.TERRA_AI_DATA_PATH, "datasets"),
        modeling=Path(settings.TERRA_AI_DATA_PATH, "modeling"),
        training=Path(settings.TERRA_AI_DATA_PATH, "training"),
    )
