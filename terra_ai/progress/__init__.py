from time import sleep
from enum import Enum
from typing import Optional, Any
from threading import Thread

from ..data.types import ConstrainedFloatValueGe0Le100
from ..data.mixins import BaseMixinData


def threading(method):
    def wrapper(*args, **kwargs):
        thread = Thread(target=method, args=args, kwargs=kwargs)
        thread.start()

    return wrapper


class PoolName(str, Enum):
    dataset_source_load = "dataset_source_load"
    dataset_choice = "dataset_choice"
    model_load = "model_load"
    training = "training"
    deploy_upload = "deploy_upload"


class ProgressData(BaseMixinData):
    percent: ConstrainedFloatValueGe0Le100 = 0
    message: str = ""
    error: str = ""
    finished: bool = True
    data: Any

    @property
    def success(self) -> bool:
        return not bool(self.error)

    def dict(self, **kwargs) -> dict:
        __data = super().dict(**kwargs)
        __data.update(
            {
                "success": self.success,
            }
        )
        return __data


class ProgressItems(BaseMixinData):
    dataset_source_load: ProgressData = ProgressData()
    dataset_choice: ProgressData = ProgressData()
    model_load: ProgressData = ProgressData()
    deploy_upload: ProgressData = ProgressData()
    training: ProgressData = ProgressData()


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
                data=kwargs.get("data", __progress.data),
            ),
        )
        print(self.__pool.training)

    def reset(self, name: PoolName, **kwargs):
        setattr(self.__pool, name, ProgressData(**kwargs))

    def monitoring(self, name: PoolName, delay: float = 1.0):
        def __output(__progress):
            print(
                "% 4i%%:" % __progress.percent,
                f"finished={__progress.finished}",
                f'error="{__progress.error}"',
            )

        sleep(delay)
        __progress = self(name)
        while not __progress.finished or not __progress.success:
            __output(__progress)
            sleep(delay)
            __progress = self(name)
        __output(__progress)


pool = ProgressPool()
