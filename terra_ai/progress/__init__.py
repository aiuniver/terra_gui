from enum import Enum
from typing import Optional
from pydantic import confloat, BaseModel
from threading import Thread


def threading(method):
    def wrapper(*args, **kwargs):
        thread = Thread(target=method, args=args, kwargs=kwargs)
        thread.start()

    return wrapper


class PoolName(str, Enum):
    dataset_source_load = "dataset_source_load"


class ProgressData(BaseModel):
    percent: confloat(ge=0, le=100) = 0
    message: str = ""
    error: str = ""
    finished: bool = False

    @property
    def success(self) -> bool:
        return not bool(self.error)

    def dict(self, *args, **kwargs) -> dict:
        __data = super().dict(*args, **kwargs)
        __data.update(
            {
                "success": self.success,
            }
        )
        return __data


class ProgressItems(BaseModel):
    dataset_source_load: ProgressData = ProgressData()


class ProgressPool:
    __pool: ProgressItems = ProgressItems()

    def __call__(self, name: PoolName, **kwargs) -> Optional[ProgressData]:
        __progress = getattr(self.__pool, name)
        if not len(kwargs.keys()):
            if __progress.finished and __progress.percent == 100:
                self.reset(name)
            return __progress

        setattr(
            self.__pool,
            name,
            ProgressData(
                percent=kwargs.get("percent", __progress.percent),
                message=kwargs.get("message", __progress.message),
                error=kwargs.get("error", __progress.error),
                finished=kwargs.get("finished", __progress.finished),
            ),
        )

    def reset(self, name: PoolName, **kwargs):
        setattr(self.__pool, name, ProgressData(**kwargs))


pool = ProgressPool()
