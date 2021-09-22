from uuid import uuid4
from typing import Optional
from pathlib import Path

from django.apps import AppConfig


class MediaConfig(AppConfig):
    name = "apps.media"

    __path_hash: dict = {}

    def path_hash(self, path: str = None, hashstr: str = None) -> Optional[str]:
        if path:
            path = Path(path)
            if not path.is_file():
                return
            hashstr = uuid4()
            self.__path_hash.update({str(hashstr): path})
            return hashstr
        if hashstr and self.__path_hash.get(hashstr):
            return self.__path_hash.pop(hashstr)
