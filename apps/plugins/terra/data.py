from dataclasses import dataclass


@dataclass
class TerraExchangeResponse:
    success: bool
    error: str
    data: dict
    stop_flag: bool

    def __init__(self, **kwargs):
        self.success = kwargs.get("success", True)
        self.error = kwargs.get("error", "")
        self.data = kwargs.get("data", {})
        self.stop_flag = kwargs.get("stop_flag", True)


@dataclass
class TerraExchangeProject:
    name: str
    hardware: str
    datasets: dict
    tags: dict
    dataset: str
    task: str
    model_name: str
    layers: dict

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "NoName")
        self.hardware = kwargs.get("hardware", "CPU")
        self.datasets = kwargs.get("datasets", {})
        self.tags = kwargs.get("tags", {})
        self.dataset = kwargs.get("dataset", "")
        self.task = kwargs.get("task", "")
        self.model_name = kwargs.get("model_name", "")
        self.layers = kwargs.get("layers", {})
