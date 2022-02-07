from typing import Any
from pydantic import validator

from ....mixins import BaseMixinData
from ...outputs import OutputsList
from ...extra import TasksRelations


class ParametersData(BaseMixinData):
    model: Any
    outputs: OutputsList = OutputsList()

    __repr_str_exclude__ = ["model"]

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"model"}})
        return super().dict(**kwargs)

    @validator("outputs", always=True)
    def _validate_outputs(cls, data, values):
        if not data:
            data = []

        _model = values.get("model")
        if not _model or not _model.layers:
            return []

        if isinstance(data, OutputsList):
            data = data.native()

        data = list(
            filter(lambda item: int(item.get("id")) in _model.outputs.ids, data)
        )

        _data = []
        for layer in _model.outputs:
            _outputs = list(
                filter(
                    lambda item: int(item.get("id")) == layer.id,
                    data,
                )
            ) + [
                {
                    "id": layer.id,
                    "classes_quantity": layer.num_classes,
                    "task": layer.task.value if layer.task else None,
                }
            ]
            _layer_data = _outputs[0]

            _task = TasksRelations.get(layer.task.value if layer.task else None)
            if not _task:
                continue

            _task_loss = (
                list(map(lambda item: item.name, _task.losses)) if _task else []
            )
            _loss = _layer_data.get("loss")
            if _loss not in _task_loss:
                _loss = _task_loss[0]
            _layer_data["loss"] = _loss

            _task_metrics = (
                list(map(lambda item: item.name, _task.metrics)) if _task else []
            )
            _metrics = list(set(_layer_data.get("metrics") or []) & set(_task_metrics))
            if not _metrics:
                _metrics = [_task_metrics[0]] if _task_metrics else []
            _layer_data["metrics"] = _metrics
            _data.append(_layer_data)

        return OutputsList(_data)
