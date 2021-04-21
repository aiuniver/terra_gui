import json
import requests

from django.conf import settings

from .data import TerraExchangeResponse, TerraExchangeProject
from .exceptions import TerraExchangeException


class TerraExchange:
    __project: TerraExchangeProject = TerraExchangeProject()

    @property
    def project(self) -> TerraExchangeProject:
        return self.__project

    @project.setter
    def project(self, props: dict):
        self.__project = TerraExchangeProject(**props)

    @property
    def api_url(self) -> str:
        return settings.TERRA_AI_EXCHANGE_API_URL

    def __get_api_url(self, name: str) -> str:
        return f"{self.api_url}/{name}/"

    def __request_get(self, *args, **kwargs) -> TerraExchangeResponse:
        if len(args) != 1:
            raise TerraExchangeException(
                "«__request_get» method must contain method name as first argument"
            )
        try:
            response = requests.get(self.__get_api_url(args[0]))
            return self.__response(response)
        except requests.exceptions.ConnectionError as error:
            return TerraExchangeResponse(success=False, error=error)
        except json.JSONDecodeError as error:
            return TerraExchangeResponse(success=False, error=error)

    def __request_post(self, *args, **kwargs) -> TerraExchangeResponse:
        if len(args) != 1:
            raise TerraExchangeException(
                "«__request_post» method must contain method name as first argument"
            )
        try:
            response = requests.post(self.__get_api_url(args[0]), json=kwargs)
            return self.__response(response)
        except requests.exceptions.ConnectionError as error:
            return TerraExchangeResponse(success=False, error=error)
        except json.JSONDecodeError as error:
            return TerraExchangeResponse(success=False, error=error)

    def __response(self, response: requests.models.Response) -> TerraExchangeResponse:
        if response.ok:
            return TerraExchangeResponse(**response.json())
        else:
            return TerraExchangeResponse(
                success=False, error=response.json().get("detail")
            )

    def call(self, *args, **kwargs) -> TerraExchangeResponse:
        if len(args) != 1:
            raise TerraExchangeException(
                "«call» method must contain method name as first argument"
            )

        name = args[0]
        method = getattr(self, f"_call_{name}", None)
        if method:
            return method(**kwargs)
        else:
            raise TerraExchangeException(f"You call undefined method «{name}»")

    def _call_get_state(self, task: str = "") -> TerraExchangeResponse:
        return self.__request_post("get_state", task=task)

    def _call_set_project_name(self, name: str) -> TerraExchangeResponse:
        self.__project.name = name
        return TerraExchangeResponse()

    def _call_prepare_dataset(self, dataset: str, task: str) -> TerraExchangeResponse:
        response = self.__request_post(
            "prepare_dataset", dataset_name=dataset, task_type=task
        )
        if response.success:
            self.__project.dataset = dataset
            self.__project.task = task
            response.data = {"dataset": dataset, "task": task}
        return response

    def _call_get_data(self) -> TerraExchangeResponse:
        return self.__request_post("get_data")

    def _call_get_models(self) -> TerraExchangeResponse:
        return self.__request_post("get_models")

    def _call_get_model_from_list(self, model_file: str) -> TerraExchangeResponse:
        return self.__request_post("get_model_from_list", model_name=model_file)

    def _call_set_model(self, layers: dict, schema: list) -> TerraExchangeResponse:
        self.__project.layers = layers
        self.__project.schema = schema
        return TerraExchangeResponse(data={"layers": layers, "schema": schema})

    def _call_set_input_layer(self) -> TerraExchangeResponse:
        response = self.__request_post("set_input_layer")
        self.__project.layers = response.data.get("layers")
        return response

    def _call_set_any_layer(self, layer_type: str = "any") -> TerraExchangeResponse:
        response = self.__request_post("set_any_layer", layer_type=layer_type)
        self.__project.layers = response.data.get("layers")
        return response

    def _call_get_change_validation(self, layers: dict) -> TerraExchangeResponse:
        response = self.__request_post("get_change_validation", layers=layers)
        self.__project.layers = response.data.get("layers")
        return response

    def _call_get_optimizer_kwargs(self, optimizer: str) -> TerraExchangeResponse:
        return self.__request_post("get_optimizer_kwargs", optimizer_name=optimizer)

    def _call_set_callbacks_switches(self, **kwargs) -> TerraExchangeResponse:
        callbacks = self.__project.callbacks
        for callback in kwargs.items():
            callbacks[callback[0]]["value"] = callback[1]
        self.__project.callbacks = callbacks
        return TerraExchangeResponse(data={"callbacks": callbacks})

    def _call_start_nn_train(self, **kwargs) -> TerraExchangeResponse:
        return self.__request_post("start_nn_train", **kwargs)

    def _call_start_evaluate(self, **kwargs) -> TerraExchangeResponse:
        return self.__request_post("start_evaluate", **kwargs)
