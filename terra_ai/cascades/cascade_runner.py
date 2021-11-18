import json
import os

import pandas as pd
from pathlib import Path

from terra_ai.cascades.common import decamelize
from terra_ai.cascades.create import json2cascade
from terra_ai.data.cascades.blocks.extra import BlockFunctionGroupChoice, FunctionParamsChoice
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice


class CascadeRunner:

    def start_cascade(self, cascade_data: CascadeDetailsData, path: Path):
        script_name, model = self._get_task_type(cascade_data=cascade_data, training_path=path)
        dataset_path = os.path.join(path, model, "model", "dataset")

        with open(os.path.join(dataset_path, "config.json"), "r", encoding="utf-8") as dataset_config:
            dataset_config_data = json.load(dataset_config)

        model_task = list(set([val.get("task") for key, val in dataset_config_data.get("outputs").items()]))[0]
        cascade_config = self._create_config(cascade_data=cascade_data, model_task=model_task,
                                             dataset_data=dataset_config_data)
        print(cascade_config)
        script_name, model = self._get_task_type(cascade_data=cascade_data, training_path=path)

        main_block = json2cascade(path=os.path.join(path, model), cascade_config=cascade_config, mode="run")
        sources = self._get_sources(dataset_path=dataset_path)
        i = 0
        if not sources:
            dataset_path = "F:\\tmp\\terraai\\datasets\\loaded\\terra\\chess_v3"
            sources = self._get_sources(dataset_path=dataset_path)
        for source in sources:
            print(source)
            if "text" in script_name:
                with open("test.txt", "w", encoding="utf-8") as f:
                    f.write(source)
                input_path = "test.txt"
            else:
                input_path = os.path.join(dataset_path, source)
                print(input_path)
            if "segmentation" in script_name or "object_detection" in script_name:
                output_path = f"F:\\tmp\\ttt\\source{i}.jpg"
                print(output_path)
                main_block(input_path=input_path, output_path=output_path)
            else:
                main_block(input_path=input_path)
            print(main_block.out)
            i += 1
        return cascade_config

    @staticmethod
    def _get_task_type(cascade_data: CascadeDetailsData, training_path: Path):
        model = "__current"
        for block in cascade_data.blocks:
            if block.group == BlockGroupChoice.Model:
                model = block.parameters.main.path
                break
        with open(os.path.join(training_path, model, "config.json"),
                  "r", encoding="utf-8") as training_config:
            training_details = json.load(training_config)
        deploy_type = training_details.get("base").get("architecture").get("type")
        return decamelize(deploy_type), model

    def _create_config(self, cascade_data: CascadeDetailsData, model_task: str, dataset_data: dict):

        classes = list(dataset_data.get("outputs").values())[0].get("classes_names")

        config = {"cascades": {}}
        adjacency_map = {}
        block_description = {}
        for block in cascade_data.blocks:
            if block.group == BlockGroupChoice.InputData:
                block_description = {
                    "input":
                        {
                            "tag": "input",
                            "type": block.parameters.main.type.value.lower()
                        },
                }
            elif block.group == BlockGroupChoice.OutputData:
                if block.parameters.main.type == BlockFunctionGroupChoice.ObjectDetection:
                    block_description = {
                        "saving": {
                            "tag": "output",
                            "type": block.parameters.main.type.value.lower(),
                            "params": {key: val for key, val in block.parameters.main.native().items()
                                       if key not in ["type"]}
                        }
                    }
                    adjacency_map.update({"saving": self._get_bind_names(cascade_data=cascade_data,
                                                                         blocks_ids=block.bind.up)})
            elif block.group == BlockGroupChoice.Model:
                block_description = {
                    "model": {
                        "tag": "model",
                        "task": model_task,
                        "model": block.parameters.main.path
                    }
                }
                adjacency_map.update({"model": self._get_bind_names(cascade_data=cascade_data,
                                                                    blocks_ids=block.bind.up)})
            else:
                _tag = block.group.value
                _task = block.parameters.main.group.value
                _name = decamelize(block.parameters.main.type.value)

                if block.group == BlockGroupChoice.Function:
                    _tag = block.group.value.lower()
                    block_parameters = FunctionParamsChoice.get_parameters(input_block=block.parameters.main.type)
                    parameters = {
                        key: val for key, val in block.parameters.main.native().items()
                        if key in block_parameters
                    }
                    if "classes" in parameters.keys() and not parameters.get("classes"):
                        parameters["classes"] = classes
                elif block.group == BlockGroupChoice.Service:
                    _tag = block.group.value.lower()
                    _task = block.parameters.main.group.value.lower()
                    _name = block.parameters.main.type.value
                    block_parameters = FunctionParamsChoice.get_parameters(input_block=block.parameters.main.type)
                    parameters = {
                        key: val for key, val in block.parameters.main.native().items()
                        if key in block_parameters
                    }
                else:
                    parameters = {
                            key: val for key, val in block.parameters.main.native().items()
                            if key not in ["type", "group"]
                        }
                block_description = {
                    block.name: {
                        "tag": _tag,
                        "task": _task,
                        "name": _name,
                        "params": parameters
                    }
                }
                adjacency_map.update({block.name: self._get_bind_names(cascade_data=cascade_data,
                                                                       blocks_ids=block.bind.up)})

            config["cascades"].update(block_description)
        config.update({"adjacency_map": adjacency_map})
        return config

    @staticmethod
    def _get_bind_names(blocks_ids: list, cascade_data: CascadeDetailsData):
        mapping = []
        _input = None
        for block in cascade_data.blocks:
            if block.id in blocks_ids:
                if block.group == BlockGroupChoice.InputData:
                    _input = "INPUT"
                elif block.group == BlockGroupChoice.Model:
                    mapping.append("model")
                else:
                    mapping.append(block.name)
        if _input:
            mapping.append(_input)
        return mapping

    @staticmethod
    def _get_sources(dataset_path: str) -> list:
        out = []
        try:
            sources = pd.read_csv(os.path.join(dataset_path, "instructions", "tables", "val.csv"))
        except Exception as e:
            return out
        for column in sources.columns:
            if column.split("_")[-1].title() in ["Image", "Text", "Audio", "Video"]:
                out = sources[column].to_list()
        return out
