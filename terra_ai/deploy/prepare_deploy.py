import json
import os
import shutil
from pathlib import Path
from copy import copy

import numpy as np
from tensorflow.keras.models import load_model

from terra_ai.callbacks.utils import YOLO_ARCHITECTURE
from terra_ai.cascades.common import decamelize
from terra_ai.data.datasets.dataset import DatasetData, DatasetOutputsData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.training.extra import ArchitectureChoice
from .postprocessing import postprocess_results
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.deploy.create_deploy_package import CascadeCreator
from terra_ai.exceptions.deploy import MethodNotImplementedException, DatasetCreateException, \
    DatasetPrepareException, ModelCreateException, PredictionException, NoPredictException, PresetsException, \
    NoTrainedModelException
from terra_ai.training import GUINN
from terra_ai.training.terra_models import BaseTerraModel, YoloTerraModel
from terra_ai.settings import DEPLOY_PATH
from ..data.deploy.extra import DeployTypeChoice
from ..exceptions.base import TerraBaseException
from ..exceptions.training import TrainingException


class DeployCreator:

    def get_deploy(self, training_path: Path, dataset: DatasetData, deploy_path: Path, page: dict):
        method_name = "get_deploy"
        try:
            model_path = Path(os.path.join(training_path, page.get("name"), "model"))
            presets_path = os.path.join(DEPLOY_PATH, "deploy_presets")

            if page.get("type") == "model":
                self._check_model_on_training(model_path=model_path)
                if os.path.exists(DEPLOY_PATH):
                    shutil.rmtree(DEPLOY_PATH, ignore_errors=True)
                    os.makedirs(DEPLOY_PATH, exist_ok=True)
                if not os.path.exists(presets_path):
                    os.makedirs(presets_path, exist_ok=True)

                with open(os.path.join(training_path, page.get("name"), "config.json"),
                          "r", encoding="utf-8") as training_config:
                    training_details = json.load(training_config)

                dataset_config_data = dataset.native()
                dataset_config_data.update({"path": dataset.path})
                deploy_type = training_details.get("base").get("architecture").get("type")

                if not dataset_config_data.get("architecture") or \
                        dataset_config_data.get("architecture") == ArchitectureChoice.Basic:
                    dataset_config_data = self._set_deploy_type(dataset_config_data)
                try:
                    dataset_data = DatasetData(**dataset_config_data)
                except Exception as error:
                    raise DatasetCreateException(
                        self.__class__.__name__, method_name
                    ).with_traceback(error.__traceback__)

                if not os.path.exists(os.path.join(deploy_path, "deploy_presets")):
                    os.mkdir(os.path.join(deploy_path, "deploy_presets"))

                try:
                    dataset = self._prepare_dataset(dataset_data=dataset_data)
                except Exception as error:
                    raise DatasetPrepareException(
                        self.__class__.__name__, method_name
                    ).with_traceback(error.__traceback__)

                try:
                    model = self._prepare_model(model_path=model_path, deploy_type=deploy_type, dataset=dataset)
                except Exception as error:
                    raise ModelCreateException(
                        self.__class__.__name__, method_name
                    ).with_traceback(error.__traceback__)

                try:
                    if dataset.data.use_generator:
                        if "Yolo" in deploy_type:
                            predict = []
                            for inp, out, serv in dataset.dataset.get('val').batch(
                                    training_details.get("base").get("batch")):
                                batch = model(inp)
                                if len(predict):
                                    for i, y in enumerate(predict):
                                        predict[i] = np.concatenate((y, batch[i]), axis=0)
                                else:
                                    predict = copy(batch)
                        else:
                            predict = model.predict(dataset.dataset.get('val').batch(1), batch_size=1)
                    else:
                        predict = model.predict(dataset.X.get('val'),
                                                batch_size=training_details.get("base").get("batch"))

                    if "Yolo" in deploy_type:
                        predict = [predict[1], predict[3], predict[5]]
                    if not len(predict):
                        raise NoPredictException(self.__class__.__name__, method_name)
                except Exception as error:
                    raise PredictionException(
                        self.__class__.__name__, method_name
                    ).with_traceback(error.__traceback__)
                try:
                    presets = self._get_presets(predict=predict, dataset_data=dataset_data,
                                                dataset=dataset, deploy_path=DEPLOY_PATH)
                except Exception as error:
                    if issubclass(error.__class__, TerraBaseException):
                        raise error
                    else:
                        raise PresetsException(
                            self.__class__.__name__, method_name
                        ).with_traceback(error.__traceback__)

                if "Dataframe" in deploy_type:
                    self._create_form_data_for_dataframe_deploy(deploy_path=deploy_path,
                                                                dataset=dataset, dataset_data=dataset_data)

                self._create_cascade(presets=presets, dataset=dataset, dataset_data=dataset_data,
                                     deploy_path=deploy_path, model_path=model_path, deploy_type=deploy_type)

                deploy_data = self._prepare_deploy(presets=presets, dataset=dataset,
                                                   deploy_path=deploy_path, model_path=model_path,
                                                   deploy_type=deploy_type)
            else:
                with open(os.path.join(DEPLOY_PATH,
                                       "deploy_presets",
                                       "presets_config.json"), "r", encoding="utf-8") as presets_config:
                    deploy_data = json.load(presets_config)

                cascade = CascadeCreator()
                cascade.copy_config(
                    deploy_path=Path(deploy_path),
                    config_path=Path(os.path.join(DEPLOY_PATH,
                                                  "deploy_presets",
                                                  "cascade_config.json"))
                )
                cascade.copy_package(
                    deploy_path=Path(deploy_path),
                    model_path=model_path
                )
                cascade.copy_script(
                    deploy_path=Path(deploy_path),
                    function_name=DeployTypeChoice(deploy_data.get("type")).demo
                )

            deploy_data.update({"page": page})

            return DeployData(**deploy_data)
        except Exception as error:
            raise error

    @staticmethod
    def _prepare_dataset(dataset_data: DatasetData) -> PrepareDataset:
        prepared_dataset = PrepareDataset(data=dataset_data, datasets_path=dataset_data.path)
        prepared_dataset.prepare_dataset()
        return prepared_dataset

    @staticmethod
    def _prepare_model(model_path: Path, deploy_type: Path, dataset: PrepareDataset):

        weight = None
        out_model = None

        if os.path.exists(os.path.join(model_path, "trained_model.trm")):
            model = load_model(os.path.join(model_path, "trained_model.trm"))
        else:
            model = BaseTerraModel(model=None,
                                   model_name="trained_model",
                                   model_path=model_path)
            model.load()
            if dataset.data.architecture in YOLO_ARCHITECTURE:
                options = GUINN().get_yolo_init_parameters(dataset=dataset)
                model = YoloTerraModel(model=None,
                                       model_name="trained_model",
                                       model_path=model_path,
                                       **options)

        for i in os.listdir(model_path):
            if i[-3:] == '.h5' and 'best' in i:
                weight = i
            if not weight:
                if i[-3:] == '.h5':
                    weight = i
        if weight:
            out_model = model.base_model
            out_model.load_weights(os.path.join(model_path, weight))
            if "Yolo" in deploy_type:
                out_model = model.yolo_model

        return out_model

    @staticmethod
    def _get_presets(predict, dataset: PrepareDataset,
                     dataset_data: DatasetData, deploy_path: Path):
        result = postprocess_results(array=predict,
                                     options=dataset,
                                     save_path=str(deploy_path),
                                     dataset_path=str(dataset_data.path))
        deploy_presets = []
        if result:
            deploy_presets = list(result.values())[0]
        return deploy_presets

    @staticmethod
    def _create_form_data_for_dataframe_deploy(dataset: PrepareDataset,
                                               dataset_data: DatasetData, deploy_path: Path):
        form_data = []
        with open(os.path.join(dataset_data.path, "config.json"), "r", encoding="utf-8") as dataset_conf:
            dataset_info = json.load(dataset_conf).get("columns", {})
        for inputs, input_data in dataset_info.items():
            if int(inputs) not in list(dataset.data.outputs.keys()):
                for column, column_data in input_data.items():
                    label = column
                    available = column_data.get("classes_names") if column_data.get("classes_names") else None
                    widget = "select" if available else "input"
                    input_type = "text"
                    if widget == "select":
                        table_column_data = {
                            "label": label,
                            "widget": widget,
                            "available": available
                        }
                    else:
                        table_column_data = {
                            "label": label,
                            "widget": widget,
                            "type": input_type
                        }
                    form_data.append(table_column_data)
        with open(os.path.join(deploy_path, "form.json"), "w", encoding="utf-8") as form_file:
            json.dump(form_data, form_file, ensure_ascii=False)

    def _create_cascade(self, presets, dataset: PrepareDataset, dataset_data: DatasetData,
                        deploy_type: str, model_path: Path, deploy_path: Path):
        if dataset.data.alias not in ["imdb", "boston_housing", "reuters"]:
            if "Dataframe" in deploy_type:
                self._create_form_data_for_dataframe_deploy(dataset=dataset, dataset_data=dataset_data,
                                                            deploy_path=deploy_path)
            if "Yolo" in deploy_type:
                func_name = "object_detection"
            else:
                func_name = decamelize(deploy_type)
            config = CascadeCreator()
            config.create_config(
                deploy_path=Path(deploy_path),
                model_path=Path(model_path),
                func_name=func_name
            )
            config.copy_package(
                deploy_path=Path(deploy_path),
                model_path=Path(model_path)
            )
            config.copy_model(
                deploy_path=Path(deploy_path),
                model_path=Path(model_path)
            )
            config.copy_script(
                deploy_path=Path(deploy_path),
                function_name=func_name
            )
            if deploy_type == ArchitectureChoice.TextSegmentation:
                with open(os.path.join(deploy_path, "format.txt"),
                          "w", encoding="utf-8") as format_file:
                    format_file.write(str(presets.get("color_map", "")))

    @staticmethod
    def _prepare_deploy(presets, deploy_path: Path, model_path: Path, deploy_type: str, dataset: PrepareDataset):

        cascade_data = {"deploy_path": deploy_path}
        out_presets_data = {"data": presets}

        if deploy_type == ArchitectureChoice.TextSegmentation:
            cascade_data.update({"tags_map": presets.get("color_map")})
            out_presets_data = {
                "data": presets.get("data", {}),
                "color_map": presets.get("color_map")
            }
        elif "Dataframe" in deploy_type:
            columns = []
            predict_column = ""
            for _input, input_columns in dataset.data.columns.items():
                for column_name in input_columns.keys():
                    columns.append(column_name[len(str(_input)) + 1:])
                    if input_columns[column_name].__class__ == DatasetOutputsData:
                        predict_column = column_name[len(str(_input)) + 1:]
            if deploy_type == ArchitectureChoice.DataframeRegression:
                tmp_data = list(zip(presets.get("preset"), presets.get("label")))
                tmp_deploy = [{"preset": elem[0], "label": elem[1]} for elem in tmp_data]
                out_presets_data = {"data": tmp_deploy}
            out_presets_data["columns"] = columns
            out_presets_data["predict_column"] = predict_column if predict_column else "Предсказанные значения"

        return dict([
            ("path_deploy", deploy_path),
            ("type", deploy_type),
            ("data", out_presets_data)
        ])

    @staticmethod
    def _set_deploy_type(dataset: dict) -> dict:
        inp_tasks = []
        out_tasks = []
        for key, val in dataset.get("inputs").items():
            if val.get("task") == LayerInputTypeChoice.Dataframe:
                tmp = []
                for value in dataset.get("columns")[key].values():
                    tmp.append(value.get("task"))
                unique_vals = list(set(tmp))
                if len(unique_vals) == 1 and unique_vals[0] in LayerInputTypeChoice.__dict__.keys() and unique_vals[0] \
                        in [LayerInputTypeChoice.Image, LayerInputTypeChoice.Text,
                            LayerInputTypeChoice.Audio, LayerInputTypeChoice.Video]:
                    inp_tasks.append(unique_vals[0])
                else:
                    inp_tasks.append(val.get("task"))
            else:
                inp_tasks.append(val.get("task"))
        for key, val in dataset.get("outputs").items():
            if val.get("task") == LayerOutputTypeChoice.Dataframe:
                tmp = []
                for value in dataset.get("columns")[key].values():
                    tmp.append(value.get("task"))
                unique_vals = list(set(tmp))
                if len(unique_vals) == 1 and unique_vals[0] in LayerOutputTypeChoice.__dict__.keys():
                    out_tasks.append(unique_vals[0])
                else:
                    out_tasks.append(val.get("task"))
            else:
                out_tasks.append(val.get("task"))

        inp_task_name = list(set(inp_tasks))[0] if len(set(inp_tasks)) == 1 else LayerInputTypeChoice.Dataframe
        out_task_name = list(set(out_tasks))[0] if len(set(out_tasks)) == 1 else LayerOutputTypeChoice.Dataframe

        if inp_task_name + out_task_name in ArchitectureChoice.__dict__.keys():
            deploy_type = ArchitectureChoice.__dict__[inp_task_name + out_task_name]
        elif out_task_name in ArchitectureChoice.__dict__.keys():
            deploy_type = ArchitectureChoice.__dict__[out_task_name]
        elif out_task_name == LayerOutputTypeChoice.ObjectDetection:
            deploy_type = ArchitectureChoice.__dict__[
                dataset.get("instructions").get(2).get("parameters").get("model").title() +
                dataset.get("instructions").get(2).get("parameters").get("yolo").title()]
        else:
            raise MethodNotImplementedException(__method=inp_task_name + out_task_name, __class="ArchitectureChoice")

        dataset["architecture"] = deploy_type
        return dataset

    def _check_model_on_training(self, model_path):
        method_name = "check on training"
        if os.path.exists(os.path.join(model_path, "log.history")):
            with open(os.path.join(model_path, "log.history"), "r", encoding="utf-8") as log_:
                history_ = json.load(log_)
                if len(history_.get("fit_log", {}).get("epochs", [])) == 0:
                    raise NoTrainedModelException(self.__class__.__name__, method_name)
        else:
            raise NoTrainedModelException(self.__class__.__name__, method_name)
