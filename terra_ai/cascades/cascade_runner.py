import json
import os
import shutil

import moviepy.editor as moviepy_editor

from typing import List, Dict, Any
from pathlib import Path

from PIL.Image import Image
from pydantic.color import Color

from terra_ai.cascades.common import decamelize
from terra_ai.cascades.create import json2cascade
from terra_ai.data.cascades.blocks.extra import (
    BlockFunctionGroupChoice,
    FunctionParamsChoice,
)
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.datasets.extra import LayerInputTypeChoice
from terra_ai.data.deploy.extra import DeployTypeChoice
from terra_ai.settings import CASCADE_PATH


class CascadeRunner:
    def start_cascade(
        self,
        cascade_data: CascadeDetailsData,
        training_path: Path,
        sources: Dict[int, List[str]],
    ):

        presets_path = os.path.join(CASCADE_PATH, "deploy_presets")
        if os.path.exists(CASCADE_PATH):
            shutil.rmtree(CASCADE_PATH, ignore_errors=True)
            os.makedirs(CASCADE_PATH, exist_ok=True)
        if not os.path.exists(presets_path):
            os.makedirs(presets_path, exist_ok=True)

        type_, model, inputs_ids = self._get_task_type(
            cascade_data=cascade_data, training_path=training_path
        )
        print(type_)
        dataset_path = os.path.join(training_path, model, "model", "dataset")

        with open(
            os.path.join(dataset_path, "config.json"), "r", encoding="utf-8"
        ) as dataset_config:
            dataset_config_data = json.load(dataset_config)

        model_task = list(
            set(
                [
                    val.get("task")
                    for key, val in dataset_config_data.get("outputs").items()
                ]
            )
        )[0]

        cascade_config = self._create_config(
            cascade_data=cascade_data,
            model_task=model_task,
            dataset_data=dataset_config_data,
            presets_path=presets_path,
        )
        print(cascade_config)
        main_block = json2cascade(
            path=os.path.join(training_path, model),
            cascade_config=cascade_config,
            mode="run",
        )

        sources = sources.get(inputs_ids[0])
        data_type = cascade_config.get("cascades").get("input").get("type")
        print(data_type)

        presets_data = self._get_presets(
            sources=sources,
            type_=type_,
            cascade=main_block,
            source_path=Path(dataset_path),
            predict_path=CASCADE_PATH,
            data_type=data_type,
        )
        print(presets_data)

        out_data = dict(
            [
                ("path_deploy", str(CASCADE_PATH)),
                ("type", type_),
                ("data", presets_data),
            ]
        )

        with open(
            os.path.join(presets_path, "presets_config.json"), "w", encoding="utf-8"
        ) as config:
            json.dump(out_data, config)

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
        with open(
            os.path.join(training_path, model, "config.json"), "r", encoding="utf-8"
        ) as training_config:
            training_details = json.load(training_config)
        deploy_type = training_details.get("base").get("architecture").get("type")

        return DeployTypeChoice(deploy_type), model, _inputs

    def _create_config(
        self,
        cascade_data: CascadeDetailsData,
        model_task: str,
        dataset_data: dict,
        presets_path: str,
    ):

        classes = list(dataset_data.get("outputs").values())[0].get("classes_names")
        num_class = list(dataset_data.get("outputs").values())[0].get("num_classes")
        if list(dataset_data.get("outputs").values())[0].get("classes_colors"):
            classes_colors = [
                list(Color(color).as_rgb_tuple())
                for color in list(dataset_data.get("outputs").values())[0].get(
                    "classes_colors"
                )
            ]

        config = {"cascades": {}}
        adjacency_map = {}
        block_description = {}
        _type = None
        for block in cascade_data.blocks:
            if "type" in block.parameters.main.native().keys():
                _type = block.parameters.main.type.value.lower()
            if block.group == BlockGroupChoice.InputData:
                if (
                    model_task == "ObjectDetection"
                    and block.parameters.main.type == LayerInputTypeChoice.Video
                    and block.parameters.main.switch_on_frame
                ):
                    _type = "video_by_frame"
                block_description = {
                    "input": {"tag": "input", "type": _type},
                }
            elif block.group == BlockGroupChoice.OutputData:
                if model_task in ["ObjectDetection", "Segmentation"]:
                    block_description = {
                        "saving": {
                            "tag": "output",
                            "type": "Video" if _type == "video_by_frame" else _type,
                        }
                    }
                    if block.parameters.main.type == LayerInputTypeChoice.Video:
                        block_description["saving"].update(
                            {
                                "params": {
                                    key: val
                                    for key, val in block.parameters.main.native().items()
                                    if key not in ["type"]
                                }
                            }
                        )
                    adjacency_map.update(
                        {
                            "saving": self._get_bind_names(
                                cascade_data=cascade_data, blocks_ids=block.bind.up
                            )
                        }
                    )
            elif block.group == BlockGroupChoice.Model:
                block_description = {
                    "model": {
                        "tag": "model",
                        "task": model_task,
                        "model": block.parameters.main.path,
                    }
                }
                adjacency_map.update(
                    {
                        "model": self._get_bind_names(
                            cascade_data=cascade_data, blocks_ids=block.bind.up
                        )
                    }
                )
            else:
                _tag = block.group.value
                _task = block.parameters.main.group.value
                _name = decamelize(block.parameters.main.type.value)

                if block.group == BlockGroupChoice.Function:
                    _tag = block.group.value.lower()
                    block_parameters = FunctionParamsChoice.get_parameters(
                        input_block=block.parameters.main.type
                    )
                    parameters = {
                        key: val
                        for key, val in block.parameters.main.native().items()
                        if key in block_parameters
                    }
                    if "classes" in parameters.keys() and not parameters.get("classes"):
                        parameters["classes"] = classes
                    # if "num_class" in parameters.keys() and not parameters.get("num_class"):
                    #     parameters["num_class"] = num_class
                    if "classes_colors" in parameters.keys() and not parameters.get(
                        "classes_colors"
                    ):
                        parameters["classes_colors"] = classes_colors
                elif block.group == BlockGroupChoice.Service:
                    _tag = block.group.value.lower()
                    _task = block.parameters.main.group.value.lower()
                    _name = block.parameters.main.type.value
                    block_parameters = FunctionParamsChoice.get_parameters(
                        input_block=block.parameters.main.type
                    )
                    parameters = {
                        key: val
                        for key, val in block.parameters.main.native().items()
                        if key in block_parameters
                    }
                else:
                    parameters = {
                        key: val
                        for key, val in block.parameters.main.native().items()
                        if key not in ["type", "group"]
                    }
                block_description = {
                    block.name: {
                        "tag": _tag,
                        "task": _task,
                        "name": _name,
                        "params": parameters,
                    }
                }
                adjacency_map.update(
                    {
                        block.name: self._get_bind_names(
                            cascade_data=cascade_data, blocks_ids=block.bind.up
                        )
                    }
                )

            config["cascades"].update(block_description)
        config.update({"adjacency_map": adjacency_map})

        with open(
            os.path.join(presets_path, "cascade_config.json"), "w", encoding="utf-8"
        ) as cascade_config:
            json.dump(config, cascade_config)

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

    def _get_presets(
        self,
        sources: List[Any],
        type_: DeployTypeChoice,
        cascade: Any,
        source_path: Path,
        data_type: str,
        predict_path: str,
    ):

        out_data = []
        iter_ = 0
        for source in sources[:3]:
            if type_ in [DeployTypeChoice.YoloV3, DeployTypeChoice.YoloV4]:
                input_path = os.path.join(source_path, source)
                if data_type == "video_by_frame":
                    data_type = "video"
                if data_type == "image":
                    predict_file_name = f"deploy_presets/result_{iter_}.webp"
                    source_file_name = f"deploy_presets/initial_{iter_}.webp"
                else:
                    predict_file_name = f"deploy_presets/result_{iter_}.webm"
                    source_file_name = f"deploy_presets/initial_{iter_}.webm"

                output_path = os.path.join(predict_path, predict_file_name)
                self._save_web_format(
                    initial_path=input_path,
                    deploy_path=os.path.join(predict_path, source_file_name),
                    source_type=data_type,
                )
                cascade(input_path=input_path, output_path=output_path)
                out_data.append(
                    {"source": source_file_name, "predict": predict_file_name}
                )
            if type_ in [DeployTypeChoice.ImageSegmentation]:
                input_path = os.path.join(source_path, source)
                file_name = f"example_{iter_}.webm"
                output_path = os.path.join(predict_path, file_name)
                cascade(input_path=input_path, output_path=output_path)
                out_data.append({"source": source, "predict": output_path})
            iter_ += 1

        return {"data": out_data}

    @staticmethod
    def _save_web_format(initial_path: str, deploy_path: str, source_type: str):
        if source_type == "image":
            img = Image.open(initial_path)
            img.save(deploy_path, "webp")
        elif source_type == "video":
            clip = moviepy_editor.VideoFileClip(initial_path)
            clip.write_videofile(deploy_path)
