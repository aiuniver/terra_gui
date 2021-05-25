import json
import re

import requests

from django.conf import settings

from .data import (
    TerraExchangeResponse,
    TerraExchangeProject,
    LayerLocation,
    Layer,
    OutputConfig,
    TrainConfig,
)
from .exceptions import TerraExchangeException
from .neural import colab_exchange


DEFAULT_MODEL_NAME = "NoName"


class TerraExchange:
    __project: TerraExchangeProject = TerraExchangeProject()

    @property
    def project(self) -> TerraExchangeProject:
        return self.__project

    @project.setter
    def project(self, props: dict):
        project = self.project.dict()
        project.update(props)
        self.__project = TerraExchangeProject(**project)

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

    def _call_autosave_project(self):
        self.project.autosave()

    def _call_get_state(self) -> TerraExchangeResponse:
        return TerraExchangeResponse(data=colab_exchange.get_state())

    def _call_set_project_name(self, name: str) -> TerraExchangeResponse:
        self.project.name = name
        return TerraExchangeResponse()

    def _call_prepare_dataset(
        self, dataset: str, is_custom: bool = False, not_load_layers: bool = False
    ) -> TerraExchangeResponse:
        tags, dataset_name, start_layers = colab_exchange.prepare_dataset(
            dataset_name=dataset,
            source="custom" if is_custom else "",
        )
        schema = [[], []]
        for index, layer in start_layers.items():
            schema[
                int(layer.get("config").get("location_type") != LayerLocation.input)
            ].append(index)

        if not not_load_layers:
            layers = {}
            outputs = {}
            for index, layer in start_layers.items():
                layers[int(index)] = Layer(**layer)
                if layers[int(index)].config.location_type == LayerLocation.output:
                    outputs[layers[int(index)].config.dts_layer_name] = OutputConfig()
            self.project.training.outputs = outputs

            self.project.layers = layers
            self.project.layers_start = layers
            self.project.layers_schema = schema
            self.project.dataset = dataset
            self.project.model_name = DEFAULT_MODEL_NAME

        return TerraExchangeResponse(
            data={
                "layers": self.project.dict().get("layers"),
                "schema": self.project.dict().get("layers_schema"),
                "dataset": self.project.dict().get("dataset"),
                "start_layers": self.project.dict().get("layers_start"),
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
        response = self.__request_post("get_models")
        if response.success:
            response.data = list(response.data) + list(colab_exchange.get_models())
        return response

    def _call_get_model_from_list(self, model_file: str) -> TerraExchangeResponse:
        data = self.__request_post(
            "get_model_from_list",
            model_name=model_file,
            input_shape=colab_exchange.get_dataset_input_shape(),
        )
        self.project.dir.clear_modeling()
        layers = {}
        for index, layer in data.data.get("layers").items():
            layers[int(index)] = Layer(config=layer)
        for index, layer in layers.items():
            for _index in layer.config.up_link:
                layers[int(_index)].down_link.append(int(index))
        output = {}
        for index, layer in layers.items():
            output[index] = layer.dict()
        self.project.model_name = model_file
        self.project.layers_start = layers
        self.project.layers_schema = data.data.get("front_model_schema", [])
        data.data.update({"layers": output})
        return data

    def _call_set_model(self, **kwargs) -> TerraExchangeResponse:
        layers = kwargs.get("layers")
        schema = kwargs.get("schema")
        self.project.layers = {}
        if layers:
            for index, layer in layers.items():
                self.project.layers[int(index)] = Layer(**layer)
        else:
            for index, layer in self.project.dict().get("layers_start"):
                self.project.layers[int(index)] = Layer(**layer)
        return TerraExchangeResponse(
            data={
                "layers": self.project.dict().get("layers"),
                "schema": schema,
                "validated": self.project.dir.validated,
            }
        )

    def _call_save_model(self, **kwargs) -> TerraExchangeResponse:
        kwargs["name"] = kwargs.get("name", "")
        if not kwargs.get("name"):
            return TerraExchangeResponse(success=False, error="Введите название модели")
        name_match = re.match("^[a-zA-Zа-яА-Я0-9\s\_\-]+$", kwargs.get("name"))
        if not name_match:
            return TerraExchangeResponse(
                success=False,
                error="Можно использовать только латиницу, кириллицу, цифры, пробел и символы `-_`",
            )
        self.project.dir.create_preview(kwargs.get("preview"))
        success, error = colab_exchange.save_model(
            name=kwargs.get("name"), overwrite=kwargs.get("overwrite")
        )
        self.project.model_name = kwargs.get("name")
        return TerraExchangeResponse(
            success=success, error=error, data={"name": self.project.model_name}
        )

    def _call_clear_model(self) -> TerraExchangeResponse:
        self.project.dir.clear_modeling()
        self.project.layers = {}
        for index, layer in self.project.dict().get("layers_start").items():
            self.project.layers[int(index)] = Layer(**layer)
        self.project.model_name = DEFAULT_MODEL_NAME
        return TerraExchangeResponse(
            data={
                "layers": self.project.dict().get("layers"),
                "schema": self.project.dict().get("layers_schema"),
            }
        )

    def _call_save_layer(self, index: int, layer: dict) -> TerraExchangeResponse:
        self.project.layers[int(index)] = Layer(**layer)
        return TerraExchangeResponse(
            data={
                "index": int(index),
                "layers": self.project.dict().get("layers"),
            }
        )

    def _call_get_keras_code(self) -> TerraExchangeResponse:
        success, output = self.project.dir.keras_code
        if success:
            return TerraExchangeResponse(data={"code": output})
        else:
            return TerraExchangeResponse(success=success, error=output)

    def _call_get_change_validation(self) -> TerraExchangeResponse:
        self.project.dir.remove_plan()
        self.project.dir.remove_keras()
        if self.project.layers:
            configs = dict(
                map(
                    lambda value: [int(value[0]), value[1].config.dict()],
                    self.project.layers.items(),
                )
            )
            response = self.__request_post(
                "get_change_validation",
                layers=configs,
                modelling_plan=colab_exchange.get_model_plan(
                    model_name=self.project.model_name
                ),
            )
            if response.success:
                validated = (
                    len(list(filter(None, response.data.get("errors").values()))) == 0
                )
                if validated:
                    self.project.dir.create_plan(response.data.get("yaml_model"))
                    self.project.dir.create_keras(response.data.get("keras_code"))
                    self.project.model_plan = response.data.get("plan")
                return TerraExchangeResponse(
                    data={
                        "errors": response.data.get("errors"),
                        "validated": validated,
                    }
                )
            else:
                response.data.update({"validated": False})
                return response
        else:
            return TerraExchangeResponse(data={"validated": False})

    def _call_before_start_training(self, **kwargs) -> TerraExchangeResponse:
        self.project.training = TrainConfig(**kwargs)
        colab_exchange._reset_out_data()
        return self.call("get_change_validation")

    def _call_start_training(self, **kwargs) -> TerraExchangeResponse:
        model_plan = colab_exchange.get_model_plan(
            self.project.model_plan, self.project.model_name
        )
        training_data = self.project.dict().get("training")
        response = self.__request_post(
            "get_model_to_colab",
            model_plan=model_plan,
            training=training_data,
        )
        if not response.success:
            return response

        model = response.data.get("model", "")
        response = colab_exchange.start_training(
            model=model,
            pathname=self.project.dir.training,
            **training_data,
        )
        return TerraExchangeResponse(data={"data": response})

    def _call_stop_training(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.stop_training()
        return TerraExchangeResponse()

    def _call_reset_training(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.reset_training()
        return TerraExchangeResponse()

    def _call_start_evaluate(self, **kwargs) -> TerraExchangeResponse:
        return self.__request_post("start_evaluate", **kwargs)
