from typing import Any
from dataclasses import dataclass

from apps.plugins.terra import terra_exchange


@dataclass
class ExchangeData:
    success: bool
    stop_flag: bool
    data: dict
    error: str

    def __init__(self, name: str, data: [Any]):
        self.success = True
        self.stop_flag = True
        self.data = {}
        self.error = ""

        method = getattr(self, f"_execute_{name}", None)
        if method:
            try:
                response = method(**data if type(data) == dict else data)
                self.success = response.success
                self.stop_flag = response.stop_flag
                self.data = response.data
                self.error = response.error
                if name != "get_data":
                    terra_exchange.call("autosave_project")
            except Exception as error:
                self.success = False
                self.error = f"[{error.__class__.__name__}] {error}"
        else:
            self.success = False
            self.error = f"Method «{name}» is undefined"

    def _execute_autosave_project(self, **kwargs):
        return terra_exchange.call("autosave_project", **kwargs)

    def _execute_set_project_name(self, **kwargs):
        return terra_exchange.call("set_project_name", **kwargs)

    def _execute_prepare_dataset(self, **kwargs):
        return terra_exchange.call("prepare_dataset", **kwargs)

    def _execute_get_data(self, **kwargs):
        return terra_exchange.call("get_data", **kwargs)

    def _execute_get_models(self, **kwargs):
        return terra_exchange.call("get_models", **kwargs)

    def _execute_get_model_from_list(self, **kwargs):
        return terra_exchange.call("get_model_from_list", **kwargs)

    def _execute_set_model(self, **kwargs):
        return terra_exchange.call("set_model", **kwargs)

    def _execute_save_model(self, **kwargs):
        return terra_exchange.call("save_model", **kwargs)

    def _execute_clear_model(self, **kwargs):
        return terra_exchange.call("clear_model", **kwargs)

    def _execute_save_layer(self, **kwargs):
        return terra_exchange.call("save_layer", **kwargs)

    def _execute_get_change_validation(self, **kwargs):
        return terra_exchange.call("get_change_validation", **kwargs)

    def _execute_get_keras_code(self, **kwargs):
        return terra_exchange.call("get_keras_code", **kwargs)

    def _execute_before_start_training(self, **kwargs):
        return terra_exchange.call("before_start_training", **kwargs)

    def _execute_start_training(self, **kwargs):
        return terra_exchange.call("start_training", **kwargs)

    def _execute_stop_training(self, **kwargs):
        return terra_exchange.call("stop_training", **kwargs)

    def _execute_reset_training(self, **kwargs):
        return terra_exchange.call("reset_training", **kwargs)

    def _execute_start_evaluate(self, **kwargs):
        return terra_exchange.call("start_evaluate", **kwargs)
