import base64
import os
import re
import json
import shutil

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

    def _update_in_training_flag(self):
        self.project.in_training = not colab_exchange.get_training_flags().get(
            "is_trained", True
        )

    def _call_autosave_project(self):
        self.project.autosave()

    def _call_get_state(self) -> TerraExchangeResponse:
        state = colab_exchange.get_state()
        return TerraExchangeResponse(data=state)

    def _call_set_project_name(self, name: str) -> TerraExchangeResponse:
        self.project.name = name
        return TerraExchangeResponse()

    def _call_prepare_dataset(
        self, dataset: str, is_custom: bool = False, not_load_layers: bool = False
    ) -> TerraExchangeResponse:
        tags, dataset_name, start_layers = colab_exchange.prepare_dataset(
            dataset_name=dataset, source="custom_dataset" if is_custom else ""
        )
        schema = [[], []]
        for index, layer in start_layers.items():
            schema[
                int(layer.get("config").get("location_type") != LayerLocation.input)
            ].append(index)

        if not not_load_layers:
            layers = {}
            outputs = {}
            num_classes = colab_exchange.get_dataset_num_classes()
            for index, layer in start_layers.items():
                layer = Layer(**layer)
                if layer.config.location_type == LayerLocation.output:
                    layer.config.num_classes = num_classes.get(
                        layer.config.dts_layer_name, 0
                    )
                    outputs[layer.config.dts_layer_name] = OutputConfig(
                        num_classes=layer.config.num_classes
                    )
                layers[int(index)] = layer

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
        self._update_in_training_flag()
        response = colab_exchange.get_data()
        response["in_training"] = self.project.in_training
        response["user_stop_train"] = colab_exchange.get_training_flags().get(
            "user_stop_train", True
        )
        self.project.dir.save_training_output(
            {
                "plots": response.get("plots", []),
                "scatters": response.get("scatters", []),
                "images": response.get("images", []),
                "texts": response.get("texts", {}),
            }
        )
        return TerraExchangeResponse(
            data=response,
            stop_flag=response.get("stop_flag", True),
            success=response.get("success", True),
            error=response.get("errors", ""),
        )

    def _call_before_load_dataset_source(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.reset_stop_flag()
        return TerraExchangeResponse()

    def _call_before_create_dataset(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.reset_stop_flag()
        return TerraExchangeResponse()

    def _call_load_dataset(self, **kwargs) -> TerraExchangeResponse:
        response = colab_exchange.load_dataset(**kwargs)
        return TerraExchangeResponse(data=response)

    def _call_create_dataset(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.create_dataset(**kwargs)
        response = self.call("get_state")
        if response.success:
            response.data.update({"error": ""})
            data = response.data
        else:
            data = {"error": "No connection to TerraAI project"}
        self.project = data

        return TerraExchangeResponse(
            data={
                "datasets": self.project.dict().get("datasets"),
                "tags": self.project.dict().get("tags"),
            }
        )

    def _call_get_models(self) -> TerraExchangeResponse:
        response = self.__request_post("get_models")
        if response.success:
            response.data = list(
                map(lambda item: {"is_terra": True, "name": item}, response.data)
            ) + list(
                map(
                    lambda item: {"is_terra": False, "name": item},
                    colab_exchange.get_models(),
                )
            )
        return response

    def _call_get_model_from_list(
        self, model_file: str, is_terra: bool
    ) -> TerraExchangeResponse:
        if is_terra:
            data = self.__request_post(
                "get_model_from_list",
                model_name=model_file,
                input_shape=colab_exchange.get_dataset_input_shape(),
            )
        else:
            with open(
                os.path.join(self.project.gd.modeling, f"{model_file}.model"), "rb"
            ) as model_ref:
                model_bin = model_ref.read()
                data = self.__request_post(
                    "get_model_from_list",
                    model_name=model_file,
                    model_file=base64.b64encode(model_bin).decode("UTF-8"),
                    input_shape=colab_exchange.get_dataset_input_shape(),
                )
        num_classes = colab_exchange.get_dataset_num_classes()
        self.project.dir.clear_modeling()
        layers = {}
        for index, layer in data.data.get("layers").items():
            if "config" not in layer.keys():
                layer = {"config": layer}
            layer = Layer(**layer)
            if layer.config.location_type == LayerLocation.output:
                layer.config.num_classes = num_classes.get(
                    layer.config.dts_layer_name, 0
                )
            layers[int(index)] = layer
        for index, layer in layers.items():
            for _index in layer.config.up_link:
                layers[int(_index)].down_link.append(int(index))
        output = {}
        for index, layer in layers.items():
            output[index] = layer.dict()
        self.project.model_name = model_file
        data.data.update({"layers": output})
        return data

    def _call_set_model(self, **kwargs) -> TerraExchangeResponse:
        layers = kwargs.get("layers")
        schema = kwargs.get("schema")
        reset_training = kwargs.get("reset_training", False)
        self.project.layers = {}
        if layers:
            for index, layer in layers.items():
                self.project.layers[int(index)] = Layer(**layer)
        else:
            for index, layer in self.project.dict().get("layers_start"):
                self.project.layers[int(index)] = Layer(**layer)
        if reset_training:
            self.call("reset_training")
        return TerraExchangeResponse(
            data={
                "layers": self.project.dict().get("layers"),
                "schema": schema,
                "validated": self.project.dir.validated,
            }
        )

    def _call_save_model(
        self, name: str, preview: str, overwrite: bool = False
    ) -> TerraExchangeResponse:
        if not name:
            return TerraExchangeResponse(success=False, error="Введите название модели")
        name_match = re.match("^[a-zA-Zа-яА-Я0-9\s\_\-]+$", name)
        if not name_match:
            return TerraExchangeResponse(
                success=False,
                error="Можно использовать только латиницу, кириллицу, цифры, пробел и символы `-_`",
            )
        fullpath = os.path.join(self.project.gd.modeling, f"{name}.model")
        if os.path.isfile(fullpath) and not overwrite:
            return TerraExchangeResponse(
                success=False,
                error="Модель с таким названием уже существует",
            )
        self.call("get_change_validation")
        self.project.dir.create_preview(preview)
        self.project.dir.create_layers(self.project.dict().get("layers"))
        filepath = shutil.make_archive(name, "zip", self.project.dir.modeling)
        shutil.move(filepath, fullpath)
        self.project.model_name = name
        return TerraExchangeResponse(data={"name": self.project.model_name})

    def _call_clear_model(self) -> TerraExchangeResponse:
        self.project.dir.clear_modeling()
        self.project.layers = {}
        for index, layer in self.project.dict().get("layers_start").items():
            self.project.layers[int(index)] = Layer(**layer)
        self.project.model_name = DEFAULT_MODEL_NAME
        self.call("reset_training")
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
            modelling_plan = colab_exchange.get_model_plan(
                model_name=self.project.model_name
            )
            response = self.__request_post(
                "get_change_validation",
                layers=configs,
                modelling_plan=modelling_plan,
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
                        "logging": json.dumps(
                            {"layers": configs, "modelling_plan": modelling_plan},
                            indent=4,
                        ),
                    }
                )
            else:
                response.data.update(
                    {
                        "validated": False,
                        "logging": json.dumps(
                            {"layers": configs, "modelling_plan": modelling_plan},
                            indent=4,
                        ),
                    }
                )
                return response
        else:
            return TerraExchangeResponse(data={"validated": False})

    def _call_before_start_training(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.reset_stop_flag()
        output = kwargs.get("checkpoint", {}).get("monitor", {}).get("output")
        out_type = kwargs.get("checkpoint", {}).get("monitor", {}).get("out_type")
        kwargs["checkpoint"]["monitor"]["out_monitor"] = (
            kwargs.get("outputs", {}).get(output, {}).get(out_type)
        )
        if out_type == "metrics":
            kwargs["checkpoint"]["monitor"]["out_monitor"] = kwargs["checkpoint"][
                "monitor"
            ]["out_monitor"][0]
        self.project.training = TrainConfig(**kwargs)
        response = self.call("get_change_validation")
        if not response.data.get("validated"):
            colab_exchange.out_data["stop_flag"] = True
        response.data["logging"] = json.dumps(
            self.project.dict().get("training"), indent=4
        )
        self._update_in_training_flag()
        self.project.autosave()
        response.data["in_training"] = self.project.in_training
        return response

    def _call_start_training(self, **kwargs) -> TerraExchangeResponse:
        self._update_in_training_flag()
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
        colab_exchange.start_training(
            model=model,
            pathname=self.project.dir.training,
            **training_data,
        )
        self._update_in_training_flag()
        self.project.autosave()
        return TerraExchangeResponse()

    def _call_stop_training(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.stop_training()
        self._update_in_training_flag()
        self.project.autosave()
        return TerraExchangeResponse()

    def _call_get_zipfiles(self) -> TerraExchangeResponse:
        response = colab_exchange.get_zipfiles()
        return TerraExchangeResponse(data=response)

    def _call_get_auto_colors(self, **kwargs) -> TerraExchangeResponse:
        response = colab_exchange.get_auto_colors(**kwargs)
        return TerraExchangeResponse(data=response)

    def _call_reset_training(self, **kwargs) -> TerraExchangeResponse:
        colab_exchange.reset_training()
        colab_exchange._reset_out_data()
        colab_exchange.out_data["stop_flag"] = True
        self.project.in_training = False
        self.project.dir.remove_training()
        self._update_in_training_flag()
        self.project.autosave()
        return TerraExchangeResponse()

    def _call_start_evaluate(self, **kwargs) -> TerraExchangeResponse:
        return self.__request_post("start_evaluate", **kwargs)

    def _call_project_new(self, **kwargs) -> TerraExchangeResponse:
        self.project.clear()
        self.__project = TerraExchangeProject()

        response = self.call("get_state")

        if response.success:
            response.data.update({"error": ""})
            data = response.data
        else:
            data = {"error": "No connection to TerraAI project"}

        self.project = data
        return response

    def _call_project_save(
        self, name: str, overwrite: bool = False
    ) -> TerraExchangeResponse:
        if not name:
            return TerraExchangeResponse(
                success=False, error="Введите название проекта"
            )
        name_match = re.match("^[a-zA-Zа-яА-Я0-9\s\_\-]+$", name)
        if not name_match:
            return TerraExchangeResponse(
                success=False,
                error="Можно использовать только латиницу, кириллицу, цифры, пробел и символы `-_`",
            )
        self.project.name = name
        self.project.autosave()
        fullpath = os.path.join(self.project.gd.projects, f"{name}.project")
        if os.path.isfile(fullpath) and not overwrite:
            return TerraExchangeResponse(
                success=False,
                error="Проект с таким названием уже существует",
            )
        self.project.save_training_files(name=name)
        filepath = shutil.make_archive(name, "zip", settings.TERRA_AI_PROJECT_PATH)
        shutil.move(filepath, fullpath)
        return TerraExchangeResponse(data={"name": name})

    def _call_project_load(self) -> TerraExchangeResponse:
        output = []
        for filename in os.listdir(self.project.gd.projects):
            if filename.endswith(".project"):
                output.append(filename[:-8])
        return TerraExchangeResponse(data=output)

    def _call_get_project(self, name: str) -> TerraExchangeResponse:
        self.project.clear()

        fullpath = os.path.join(self.project.gd.projects, f"{name}.project")
        shutil.unpack_archive(fullpath, settings.TERRA_AI_PROJECT_PATH, "zip")

        self.__project = TerraExchangeProject()

        response = self.call("get_state")

        if response.success:
            response.data.update({"error": ""})
            data = response.data
        else:
            data = {"error": "No connection to TerraAI project"}

        self.project = data
        return response
