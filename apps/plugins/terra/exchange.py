import json
import copy
import requests

from django.conf import settings

from .data import TerraExchangeResponse, TerraExchangeProject
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

    def __prepare_layers(self, layers: dict) -> dict:
        def get_down_link_list(index):
            output = []
            for item_index, item in layers.items():
                if int(index) in item.get("config").get("up_link"):
                    output.append(int(item_index))
            return output

        for index, layer in layers.items():
            config = layer.get("config", {})
            params = config.get("params", {})
            if "id" not in layer:
                layer["id"] = int(index)
            if "index" not in layer:
                layer["index"] = int(index)
            if "type" not in layer:
                if config.get("type") == "Input":
                    layer["type"] = "input"
                else:
                    layer["type"] = "middle"
            if "down_link" not in layer:
                layer["down_link"] = get_down_link_list(index)
            param_conf = colab_exchange.layers_params.get(config.get("type"), {})
            for group_name, group in param_conf.items():
                if group_name not in params:
                    params[group_name] = {}
                for param_name, param in group.items():
                    if param_name not in params[group_name]:
                        params[group_name][param_name] = param.get("default")
        return layers

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
        return TerraExchangeResponse(data=colab_exchange.get_state(task=task))

    def _call_set_project_name(self, name: str) -> TerraExchangeResponse:
        self.__project.name = name
        return TerraExchangeResponse()

    def _call_prepare_dataset(
        self, dataset: str, task: str, is_custom: bool = False
    ) -> TerraExchangeResponse:
        tags, name, start_layers, layers_data_state = colab_exchange.prepare_dataset(
            dataset_name=dataset,
            task_type=task,
            source="custom" if is_custom else "",
        )
        if not len(start_layers.keys()):
            start_layers[1] = {
                "name": f"l1_Input",
                "type": "Input",
                "params": {"main": {}, "extra": {}},
                "up_link": [],
                "inp_shape": [],
                "out_shape": [],
            }
            start_layers[2] = {
                "name": f"l2_Dense",
                "type": "Dense",
                "params": {"main": {}, "extra": {}},
                "up_link": [],
                "inp_shape": [],
                "out_shape": [],
            }

        layers = {}
        schema = [[], []]
        for index, layer in start_layers.items():
            schema[int(layer.get("type") != "Input")].append(index)
            if not len(layer.get("params", {}).keys()):
                layer["params"] = {"main": {}, "extra": {}}
            layers[index] = {
                "id": index,
                "index": index,
                "config": layer,
                "type": "input" if int(layer.get("type") == "Input") else "output",
            }

        self.__project.layers = self.__prepare_layers(layers)
        self.__project.schema = schema
        self.__project.start_layers = start_layers
        self.__project.dataset = dataset
        self.__project.task = task
        return TerraExchangeResponse(
            data={
                "layers": self.__project.layers,
                "schema": self.__project.schema,
                "dataset": self.__project.dataset,
                "task": self.__project.task,
                "start_layers": self.__project.start_layers,
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
        data = self.__request_post("get_model_from_list", model_name=model_file)
        layers = {}
        for index, layer in data.data.get("layers").items():
            layers[index] = {"config": layer}
        data.data.update({"layers": self.__prepare_layers(layers)})
        return data

    def _call_set_model(self, layers: dict, schema: list) -> TerraExchangeResponse:
        for index, layer in layers.items():
            params = layer.get("config").get("params", None)
            params = params if params else {}
            for name, param in params.items():
                if param.get("type") == "tuple" and isinstance(
                    param.get("default"), list
                ):
                    default = list(map(lambda value: str(value), param.get("default")))
                    param.update({"default": ",".join(default)})
                    params.update({name: param})
            layer["config"].update({"params": params})
            layers[index] = layer

        if not layers:
            schema = [[], []]
            for index, layer in self.__project.start_layers.items():
                schema[int(layer.get("type") != "Input")].append(index)
                if not len(layer.get("params", {}).keys()):
                    layer["params"] = {"main": {}, "extra": {}}
                layers[index] = {
                    "id": index,
                    "index": index,
                    "config": layer,
                    "type": "input" if int(layer.get("type") == "Input") else "output",
                }

        self.__project.layers = self.__prepare_layers(layers)
        self.__project.schema = schema
        return TerraExchangeResponse(
            data={"layers": self.__project.layers, "schema": schema}
        )

    def _call_clear_model(self) -> TerraExchangeResponse:
        layers = {}
        schema = [[], []]
        for index, layer in self.__project.start_layers.items():
            schema[int(layer.get("type") != "Input")].append(index)
            if not len(layer.get("params", {}).keys()):
                layer["params"] = {"main": {}, "extra": {}}
            layers[index] = {
                "id": index,
                "index": index,
                "config": layer,
                "type": "input" if int(layer.get("type") == "Input") else "output",
            }

        self.__project.layers = self.__prepare_layers(layers)
        self.__project.schema = schema
        return TerraExchangeResponse(
            data={"layers": self.__project.layers, "schema": schema}
        )

    def _call_set_input_layer(self) -> TerraExchangeResponse:
        response = self.__request_post("set_input_layer")
        self.__project.layers = self.__prepare_layers(response.data.get("layers"))
        return response

    def _call_set_any_layer(self, layer_type: str = "any") -> TerraExchangeResponse:
        response = self.__request_post("set_any_layer", layer_type=layer_type)
        self.__project.layers = self.__prepare_layers(response.data.get("layers"))
        return response

    def _call_save_layer(self, **kwargs) -> TerraExchangeResponse:
        layers = self.__project.layers
        layers[str(kwargs.get("id"))] = kwargs
        self.__project.layers = self.__prepare_layers(layers)
        return TerraExchangeResponse(data=self.__project.layers)

    def _call_get_change_validation(self) -> TerraExchangeResponse:
        layers = {}
        for index, layer in self.__project.layers.items():
            config = copy.deepcopy(layer.get("config"))
            layers[str(index)] = config
        if layers:
            return self.__request_post("get_change_validation", layers=layers)
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
