from dataclasses import dataclass

from django.urls import reverse_lazy


UNDEFINED = "\033[0;31mundefined\033[0m"


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
    error: str
    name: str
    hardware: str
    datasets: dict
    tags: dict
    dataset: str
    task: str
    model_name: str
    layers: dict
    start_layers: dict
    schema: list
    layers_types: dict
    optimizers: list
    callbacks: dict
    path: dict

    def __init__(self, **kwargs):
        self.error = kwargs.get("error", "")
        self.name = kwargs.get("name", "NoName")
        self.hardware = kwargs.get("hardware", "CPU")
        self.datasets = kwargs.get("datasets", {})
        self.tags = kwargs.get("tags", {})
        self.dataset = kwargs.get("dataset", "")
        self.task = kwargs.get("task", "")
        self.model_name = kwargs.get("model_name", "")
        self.layers = kwargs.get("layers", {})
        self.start_layers = kwargs.get("start_layers", {})
        self.schema = kwargs.get("schema", [])
        self.layers_types = kwargs.get("layers_types", {})
        self.optimizers = kwargs.get("optimizers", [])
        self.callbacks = kwargs.get("callbacks", {})
        self.path = {
            "datasets": reverse_lazy("apps_project:datasets"),
            "modeling": reverse_lazy("apps_project:modeling"),
            "training": reverse_lazy("apps_project:training"),
        }

    def __repr__(self):
        return f"""TerraExchangeProject:
    error        : {True if self.error else False}
    name         : {self.name}
    hardware     : {self.hardware}
    datasets     : {len(self.datasets.keys())}
    tags         : {len(self.tags.keys())}
    dataset      : {self.dataset}
    task         : {self.task}
    model_name   : {self.model_name}
    layers       : {len(self.layers.keys())}
    start_layers : {len(self.start_layers.keys())}
    schema       : {len(self.schema)}
    layers_types : {len(self.layers_types.keys())}
    optimizers   : {len(self.optimizers)}
    callbacks    : {len(self.callbacks.keys())}
    path         : datasets -> {self.path.get("modeling", UNDEFINED)}
                   modeling -> {self.path.get("modeling", UNDEFINED)}
                   training -> {self.path.get("training", UNDEFINED)}"""

    @property
    def dataset_selected(self) -> bool:
        return self.dataset != "" and self.task != ""

    @property
    def as_json_string(self) -> dict:
        output = self.__dict__
        path = {}
        for name, value in self.path.items():
            path.update({name: str(value)})
        output.update({"path": path, "dataset_selected": self.dataset_selected})
        return output
