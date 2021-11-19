import json
import os
from typing import List, Dict
from pathlib import Path

from pydantic.color import Color

from terra_ai.cascades.common import decamelize
from terra_ai.cascades.create import json2cascade
from terra_ai.data.cascades.blocks.extra import BlockFunctionGroupChoice, FunctionParamsChoice
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.datasets.extra import LayerInputTypeChoice


class CascadeRunner:

    def start_cascade(self, cascade_data: CascadeDetailsData, path: Path, sources: Dict[int, List[str]]):
        script_name, model, inputs_ids = self._get_task_type(cascade_data=cascade_data, training_path=path)
        print(script_name)
        dataset_path = os.path.join(path, model, "model", "dataset")

        with open(os.path.join(dataset_path, "config.json"), "r", encoding="utf-8") as dataset_config:
            dataset_config_data = json.load(dataset_config)

        model_task = list(set([val.get("task") for key, val in dataset_config_data.get("outputs").items()]))[0]

        cascade_config = self._create_config(cascade_data=cascade_data, model_task=model_task,
                                             dataset_data=dataset_config_data)
        print(cascade_config)
        main_block = json2cascade(path=os.path.join(path, model), cascade_config=cascade_config, mode="run")

        i = 0
        presets_data = []
        for source in sources.get(inputs_ids[0])[:2]:
            print(source)
            if "text" in script_name:
                with open("test.txt", "w", encoding="utf-8") as f:
                    f.write(source)
                input_path = "test.txt"
            else:
                input_path = os.path.join(dataset_path, source)
            if "segmentation" in script_name or "object_detection" in script_name:
                output_path = f"F:\\tmp\\ttt\\source{i}.webm"
                main_block(input_path=input_path, output_path=output_path)
                presets_data.append({
                    "source": source,
                    "detection": output_path
                })
            else:
                main_block(input_path=input_path)
            print(main_block.out)
            i += 1
        print(presets_data)
        return cascade_config

    @staticmethod
    def _get_task_type(cascade_data: CascadeDetailsData, training_path: Path):
        model = "__current"
        _inputs = []
        for block in cascade_data.blocks:
            if block.group == BlockGroupChoice.Model:
                model = block.parameters.main.path
            if block.group == BlockGroupChoice.InputData:
                _inputs.append(block.id)
        with open(os.path.join(training_path, model, "config.json"),
                  "r", encoding="utf-8") as training_config:
            training_details = json.load(training_config)
        deploy_type = training_details.get("base").get("architecture").get("type")
        if "Yolo" in deploy_type:
            deploy_type = "ObjectDetection"
        return decamelize(deploy_type), model, _inputs

    def _create_config(self, cascade_data: CascadeDetailsData, model_task: str, dataset_data: dict):

        classes = list(dataset_data.get("outputs").values())[0].get("classes_names")
        num_class = list(dataset_data.get("outputs").values())[0].get("num_classes")
        if list(dataset_data.get("outputs").values())[0].get("classes_colors"):
            classes_colors = [list(Color(color).as_rgb_tuple()) for color in
                              list(dataset_data.get("outputs").values())[0].get("classes_colors")]

        config = {"cascades": {}}
        adjacency_map = {}
        block_description = {}
        _type = None
        for block in cascade_data.blocks:
            if "type" in block.parameters.main.native().keys():
                _type = block.parameters.main.type.value.lower()
            if block.group == BlockGroupChoice.InputData:
                if model_task == "ObjectDetection" and \
                        block.parameters.main.type == LayerInputTypeChoice.Video \
                        and block.parameters.main.switch_on_frame:
                    _type = "video_by_frame"
                block_description = {
                    "input":
                        {
                            "tag": "input",
                            "type": _type
                        },
                }
            elif block.group == BlockGroupChoice.OutputData:
                if model_task in ["ObjectDetection", "Segmentation"]:
                    block_description = {
                        "saving": {
                            "tag": "output",
                            "type": "Video" if _type == "video_by_frame" else _type
                        }
                    }
                    if block.parameters.main.type == LayerInputTypeChoice.Video:
                        block_description["saving"].update({
                            "params": {
                                key: val for key, val in block.parameters.main.native().items() if key not in ["type"]
                            }
                        })
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
                    # if "num_class" in parameters.keys() and not parameters.get("num_class"):
                    #     parameters["num_class"] = num_class
                    if "classes_colors" in parameters.keys() and not parameters.get("classes_colors"):
                        parameters["classes_colors"] = classes_colors
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
