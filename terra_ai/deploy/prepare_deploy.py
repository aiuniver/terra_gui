import json
import os
from pathlib import Path

from tensorflow.keras.models import load_model

from terra_ai.cascades.common import decamelize
from terra_ai.data.datasets.dataset import DatasetData, DatasetOutputsData
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.deploy.create_deploy_package import CascadeCreator
from terra_ai.training.yolo_utils import create_yolo


class DeployCreator:

    def get_deploy(self, training_path: str, deploy_path: str):

        with open(os.path.join(training_path, "config.json"), "r", encoding="utf-8") as training_config:
            training_details = json.load(training_config)

        model_path = training_details.get("model_path")
        deploy_type = training_details.get("base").get("architecture").get("type")

        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as dataset_config:
            dataset_data = DatasetData(**json.load(dataset_config))

        if not os.path.exists(os.path.join(deploy_path, "deploy_presets")):
            os.mkdir(os.path.join(deploy_path, "deploy_presets"))

        dataset = self._prepare_dataset(dataset_data=dataset_data)
        model = self._prepare_model(model_path=model_path, deploy_type=deploy_type, dataset=dataset)
        if dataset.data.use_generator:
            predict = model.predict(dataset.dataset.get('val').batch(1), batch_size=1)
        else:
            predict = model.predict(dataset.X.get('val'), batch_size=training_details.get("base").get("batch"))

        presets = self._get_presets(predict=predict, dataset_data=dataset_data,
                                    dataset=dataset, deploy_path=deploy_path)
        if "Dataframe" in training_details.base.architecture.type.value:
            self._create_form_data_for_dataframe_deploy(deploy_path=deploy_path,
                                                        dataset=dataset, dataset_data=dataset_data)

        self._create_cascade(presets=presets, dataset=dataset, dataset_data=dataset_data,
                             deploy_path=deploy_path, model_path=model_path, deploy_type=deploy_type)
        return DeployData(self._prepare_deploy(presets=presets, dataset=dataset,
                                               deploy_path=deploy_path, model_path=model_path, deploy_type=deploy_type))

    @staticmethod
    def _prepare_dataset(dataset_data: DatasetData) -> PrepareDataset:
        prepared_dataset = PrepareDataset(data=dataset_data, datasets_path=dataset_data.path)
        prepared_dataset.prepare_dataset()
        return prepared_dataset

    @staticmethod
    def _prepare_model(model_path: str, deploy_type: str, dataset: PrepareDataset):

        weight = None
        out_model = None

        model = load_model(os.path.join(model_path, "trained_model.trm"), compile=False)

        for i in os.listdir(model_path):
            if i[-3:] == '.h5' and 'best' in i:
                weight = i
        if weight:
            model.load_weights(os.path.join(model_path, weight))
            out_model = model
            if "Yolo" in deploy_type:
                out_model = create_yolo(model=model, input_size=416, channels=3, training=False,
                                        classes=dataset.data.outputs.get(2).classes_names,
                                        version=dataset.instructions.get(2).get('2_object_detection').get('yolo'))

        return out_model

    @staticmethod
    def _get_presets(predict, dataset: PrepareDataset,
                     dataset_data: DatasetData, deploy_path: str):
        result = CreateArray().postprocess_results(array=predict,
                                                   options=dataset,
                                                   save_path=deploy_path,
                                                   dataset_path=str(dataset_data.path))
        deploy_presets = []
        if result:
            deploy_presets = list(result.values())[0]
        return deploy_presets

    @staticmethod
    def _create_form_data_for_dataframe_deploy(dataset: PrepareDataset,
                                               dataset_data: DatasetData, deploy_path: str):
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
                        deploy_type: str, model_path: str, deploy_path: str):
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
            config.copy_script(
                deploy_path=Path(deploy_path),
                function_name=func_name
            )
            if deploy_type == ArchitectureChoice.TextSegmentation:
                with open(os.path.join(deploy_path, "format.txt"),
                          "w", encoding="utf-8") as format_file:
                    format_file.write(str(presets.get("tags_map", "")))

    @staticmethod
    def _prepare_deploy(presets, deploy_path: str, model_path: str, deploy_type: str, dataset: PrepareDataset):

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
            ("path", deploy_path),
            ("path_model", model_path),
            ("type", deploy_type),
            ("data", out_presets_data)
        ])
