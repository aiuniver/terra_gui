from typing import Optional, Any
from pydantic import validator

from ....mixins import BaseMixinData
from ...checkpoint import CheckpointData
from ...outputs import OutputsList
from ...extra import TasksRelations


class ParametersData(BaseMixinData):
    model: Any
    outputs: OutputsList = OutputsList()
    checkpoint: Optional[CheckpointData]

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
                    "task": str(layer.task),
                }
            ]
            _layer_data = _outputs[0]

            _task = TasksRelations.get(str(layer.task))
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

    @validator("checkpoint", always=True)
    def _validate_checkpoint(cls, data, values):
        if not data:
            data = {}

        _model = values.get("model")
        if not _model or not _model.layers:
            return None

        if isinstance(data, CheckpointData):
            data = data.native()

        _layer_id = int(data.get("layer", 0))
        _layer = (
            _model.outputs.get(_layer_id)
            if _layer_id in _model.outputs.ids
            else _model.outputs[0]
        )
        data["layer"] = _layer.id

        _task = TasksRelations.get(str(_layer.task))
        if not _task:
            return None
        _task_metrics = (
            list(map(lambda item: item.name, _task.metrics)) if _task else []
        )
        if data.get("metric_name") not in _task_metrics:
            data["metric_name"] = _task_metrics[0]

        return CheckpointData(**data)
