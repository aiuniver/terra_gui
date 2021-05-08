import json
import requests

from django.conf import settings

from .data import (
    TerraExchangeResponse,
    TerraExchangeProject,
    LayerLocation,
    LayerDict,
    Layer,
)
from .exceptions import TerraExchangeException
from .neural import colab_exchange


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
            return TerraExchangeResponse(success=False, error=str(error))
        except json.JSONDecodeError as error:
            return TerraExchangeResponse(success=False, error=str(error))

    def __request_post(self, *args, **kwargs) -> TerraExchangeResponse:
        if len(args) != 1:
            raise TerraExchangeException(
                "«__request_post» method must contain method name as first argument"
            )
        try:
            response = requests.post(self.__get_api_url(args[0]), json=kwargs)
            return self.__response(response)
        except requests.exceptions.ConnectionError as error:
            return TerraExchangeResponse(success=False, error=str(error))
        except json.JSONDecodeError as error:
            return TerraExchangeResponse(success=False, error=str(error))

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

    def _call_get_state(self) -> TerraExchangeResponse:
        return TerraExchangeResponse(data=colab_exchange.get_state())

    def _call_set_project_name(self, name: str) -> TerraExchangeResponse:
        self.__project.name = name
        return TerraExchangeResponse()

    def _call_prepare_dataset(
        self, dataset: str, is_custom: bool = False
    ) -> TerraExchangeResponse:
        tags, dataset_name, start_layers = colab_exchange.prepare_dataset(
            dataset_name=dataset,
            source="custom" if is_custom else "",
        )
        schema = [[], []]
        for index, layer in start_layers.items.items():
            schema[int(layer.config.location_type != LayerLocation.input)].append(index)

        self.__project.layers = start_layers
        self.__project.start_layers = start_layers
        self.__project.schema = schema
        self.__project.dataset = dataset
        return TerraExchangeResponse(
            data={
                "layers": self.__project.layers.as_dict.get("items"),
                "schema": self.__project.schema,
                "dataset": self.__project.dataset,
                "start_layers": self.__project.start_layers.as_dict.get("items"),
            }
        )

    def _call_get_data(self) -> TerraExchangeResponse:
        response = colab_exchange.get_data()
        return TerraExchangeResponse(
            data=response,
            stop_flag=response.get("stop_flag", True),
            success=response.get("success", True),
        )

    def _call_get_models(self) -> TerraExchangeResponse:
        return self.__request_post("get_models")

    def _call_get_model_from_list(self, model_file: str) -> TerraExchangeResponse:
        data = self.__request_post(
            "get_model_from_list",
            model_name=model_file,
            input_shape=colab_exchange.get_dataset_input_shape(),
        )
        layers = LayerDict()
        for index, layer in data.data.get("layers").items():
            layers.items[int(index)] = Layer(config=layer)
        for index, layer in layers.items.items():
            for _index in layer.config.up_link:
                layers.items[int(_index)].down_link.append(int(index))
        data.data.update({"layers": layers.as_dict.get("items")})
        return data

    def _call_set_model(self, **kwargs) -> TerraExchangeResponse:
        layers = kwargs.get("layers")
        schema = kwargs.get("schema")
        if layers:
            self.__project.layers = LayerDict()
            for index, layer in layers.items():
                self.__project.layers.items[int(index)] = Layer(**layer)
        else:
            self.__project.layers = self.__project.start_layers
        return TerraExchangeResponse(
            data={
                "layers": self.__project.layers.as_dict.get("items"),
                "schema": schema,
            }
        )

    def _call_clear_model(self) -> TerraExchangeResponse:
        self.__project.layers = self.__project.start_layers
        return TerraExchangeResponse(
            data={
                "layers": self.__project.layers.as_dict.get("items"),
                "schema": self.__project.schema,
            }
        )

    def _call_save_layer(self, index: int, layer: dict) -> TerraExchangeResponse:
        self.__project.layers.items[int(index)] = Layer(**layer)
        return TerraExchangeResponse(
            data={
                "index": int(index),
                "layers": self.__project.layers.as_dict.get("items"),
            }
        )

    def _call_get_change_validation(self) -> TerraExchangeResponse:
        layers = self.__project.layers
        if layers:
            configs = dict(
                map(
                    lambda value: [int(value[0]), value[1].get("config")],
                    layers.as_dict.get("items").items(),
                )
            )
            return self.__request_post("get_change_validation", layers=configs)
        else:
            return TerraExchangeResponse()

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
