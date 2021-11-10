from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.presets.training import TasksGroups
from terra_ai.data.training.train import DEFAULT_TRAINING_PATH_NAME


def correct_training(data: dict, model: ModelDetailsData):
    if not data.get("base"):
        data["base"] = {}
    if not data["base"].get("architecture"):
        data["base"]["architecture"] = {}
    data["base"]["architecture"].update({"model": model})
    if not data["base"]["architecture"].get("parameters"):
        data["base"]["architecture"]["parameters"] = {}
    if not data["base"]["architecture"]["parameters"].get("outputs"):
        data["base"]["architecture"]["parameters"]["outputs"] = []
    _outputs = (
        data.get("base", {})
        .get("architecture", {})
        .get("parameters", {})
        .get("outputs", [])
    )
    _outputs_correct = []
    for _output in _outputs:
        _metrics = _output.get("metrics", [])
        _loss = _output.get("loss", "")
        _task = _output.get("task")
        if not _task:
            _metrics = []
            _loss = ""
        else:
            _task_groups = list(
                filter(lambda item: item.get("task") == _task, TasksGroups)
            )
            _task_group = _task_groups[0] if len(_task_groups) else None
            if _task_group:
                _metrics = list(set(_metrics) & set(_task_group.get("metrics")))
                if not len(_metrics):
                    _metrics = [_task_group.get("metrics")[0].value]
                if _loss not in _task_group.get("losses"):
                    _loss = _task_group.get("losses")[0].value
            else:
                _metrics = []
                _loss = ""
        _output["metrics"] = _metrics
        _output["loss"] = _loss
        _outputs_correct.append(_output)
    data["base"]["architecture"]["parameters"]["outputs"] = _outputs_correct
    _checkpoint = _outputs = (
        data.get("base", {})
        .get("architecture", {})
        .get("parameters", {})
        .get("checkpoint", {})
    )
    if _checkpoint:
        _layer = _checkpoint.get("layer")
        _metric_name = _checkpoint.get("metric_name")
        _outputs = list(filter(lambda item: item.get("id") == _layer, _outputs_correct))
        _output = _outputs[0] if len(_outputs) else None
        if _output:
            if _metric_name not in _output.get("metrics"):
                _metric_name = (
                    _output.get("metrics")[0] if len(_output.get("metrics")) else ""
                )
        else:
            _layer = ""
            _metric_name = ""
        _checkpoint["layer"] = _layer
        _checkpoint["metric_name"] = _metric_name
        data["base"]["architecture"]["parameters"]["checkpoint"] = _checkpoint
    data["interactive"] = {}
    data["model"] = model
    data["name"] = data.get("name", DEFAULT_TRAINING_PATH_NAME)
    return data
