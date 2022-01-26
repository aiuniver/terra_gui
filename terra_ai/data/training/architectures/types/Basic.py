from typing import Optional
from pydantic import validator

from ...checkpoint import CheckpointData
from ...extra import TasksRelations
from . import Base


class ParametersData(Base.ParametersData):
    checkpoint: Optional[CheckpointData]

    @validator("checkpoint", always=True)
    def _validate_checkpoint(cls, data, values, field):
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
            else (_model.outputs[0] if _model.outputs else None)
        )
        if not _layer:
            return None

        data["layer"] = _layer.id

        _task = TasksRelations.get(_layer.task.value if _layer.task else None)
        if not _task:
            return None
        _task_metrics = (
            list(map(lambda item: item.name, _task.metrics)) if _task else []
        )
        if data.get("metric_name") not in _task_metrics:
            data["metric_name"] = _task_metrics[0]

        return CheckpointData(**data)
