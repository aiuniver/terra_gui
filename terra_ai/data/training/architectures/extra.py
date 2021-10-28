from typing import Any
from pydantic import PositiveFloat, validator

from ...mixins import BaseMixinData
from ..outputs import OutputsList
from ..extra import TasksRelations


class YoloParameters(BaseMixinData):
    train_lr_init: PositiveFloat = 1e-4
    train_lr_end: PositiveFloat = 1e-6
    yolo_iou_loss_thresh: PositiveFloat = 0.5
    train_warmup_epochs: PositiveFloat = 2


class OutputsParametersData(BaseMixinData):
    model: Any
    outputs: OutputsList = OutputsList()

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
                    "task": layer.task.value,
                }
            ]
            print(
                2,
                list(
                    filter(
                        lambda item: int(item.get("id")) == layer.id,
                        data,
                    )
                ),
            )
            _layer_data = _outputs[0]

            _task = TasksRelations.get(layer.task.value)

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
            if _metrics not in _task_metrics:
                _metrics = [_task_metrics[0]]
            _layer_data["metrics"] = _metrics
            data.append(_layer_data)

        print(1, data)
        return OutputsList(data)
