from typing import Any
from dataclasses import dataclass

from apps.plugins.terra import terra_exchange
from apps.plugins.terra.data import TerraExchangeResponse
from apps.plugins.terra.utils import get_traceback_text

from . import serializers


@dataclass
class ExchangeData:
    success: bool
    stop_flag: bool
    data: dict
    error: str
    tb: list

    def __init__(self, name: str, data: [Any]):
        self.success = True
        self.stop_flag = True
        self.data = {}
        self.error = ""
        self.tb = []

        method = getattr(self, f"_execute_{name}", None)
        if method:
            try:
                response = method(**data if type(data) == dict else data)
                self.success = response.success
                self.stop_flag = response.stop_flag
                self.data = response.data
                self.error = response.error
                self.tb = response.tb
                if name not in ["get_data"]:
                    terra_exchange.call("autosave_project")
            except Exception as error:
                self.tb = get_traceback_text(error.__traceback__)
                self.success = False
                self.error = f"[{error.__class__.__name__}] {error}"
        else:
            self.success = False
            self.error = f"Method «{name}» is undefined"

    def _response_error(self, message: str) -> TerraExchangeResponse:
        return TerraExchangeResponse(success=False, error=str(message))

    def _execute_autosave_project(self):
        return terra_exchange.call("autosave_project")

    def _execute_set_project_name(self, **kwargs):
        serializer = serializers.SetProjectNameSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("set_project_name", **serializer.validated_data)

    def _execute_get_datasets_info(self, **kwargs):
        return terra_exchange.call("get_datasets_info", **kwargs)

    def _execute_prepare_dataset(self, **kwargs):
        serializer = serializers.PrepareDatasetSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("prepare_dataset", **serializer.validated_data)

    def _execute_get_auto_colors(self, **kwargs):
        serializer = serializers.GetAutoColorsSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("get_auto_colors", **serializer.validated_data)

    def _execute_before_load_dataset_source(self):
        return terra_exchange.call("before_load_dataset_source")

    def _execute_before_create_dataset(self):
        return terra_exchange.call("before_create_dataset")

    def _execute_load_dataset(self, **kwargs):
        return terra_exchange.call("load_dataset", **kwargs)

    def _execute_create_dataset(self, **kwargs):
        return terra_exchange.call("create_dataset", **kwargs)

    def _execute_remove_dataset(self, **kwargs):
        serializer = serializers.RemoveDatasetSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("remove_dataset", **serializer.validated_data)

    def _execute_remove_model(self, **kwargs):
        serializer = serializers.RemoveModelSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("remove_model", **serializer.validated_data)

    def _execute_get_data(self):
        return terra_exchange.call("get_data")

    def _execute_get_models(self):
        return terra_exchange.call("get_models")

    def _execute_get_model_from_list(self, **kwargs):
        serializer = serializers.GetModelFromListSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("get_model_from_list", **serializer.validated_data)

    def _execute_set_model(self, **kwargs):
        serializer = serializers.SetModelSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("set_model", **serializer.validated_data)

    def _execute_save_model(self, **kwargs):
        serializer = serializers.SaveModelSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("save_model", **serializer.validated_data)

    def _execute_clear_model(self):
        return terra_exchange.call("clear_model")

    def _execute_save_layer(self, **kwargs):
        serializer = serializers.SaveLayerSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("save_layer", **serializer.validated_data)

    def _execute_get_change_validation(self):
        return terra_exchange.call("get_change_validation")

    def _execute_get_keras_code(self):
        return terra_exchange.call("get_keras_code")

    def _execute_before_start_training(self, **kwargs):
        serializer = serializers.BeforeStartTrainingSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("before_start_training", **serializer.validated_data)

    def _execute_start_training(self):
        return terra_exchange.call("start_training")

    def _execute_stop_training(self):
        return terra_exchange.call("stop_training")

    def _execute_reset_training(self):
        return terra_exchange.call("reset_training")

    def _execute_start_evaluate(self):
        return terra_exchange.call("start_evaluate")

    def _execute_project_new(self):
        return terra_exchange.call("project_new")

    def _execute_project_save(self, **kwargs):
        serializer = serializers.ProjectSaveSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("project_save", **serializer.validated_data)

    def _execute_project_load(self):
        return terra_exchange.call("project_load")

    def _execute_get_project(self, **kwargs):
        serializer = serializers.GetProjectSerializer(data=kwargs)
        if not serializer.is_valid():
            return self._response_error(str(serializer.errors))
        return terra_exchange.call("get_project", **serializer.validated_data)
