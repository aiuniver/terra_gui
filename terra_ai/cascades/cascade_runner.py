import json
import os
import shutil
import numpy as np
from copy import copy
import moviepy.editor as moviepy_editor
from pydub import AudioSegment

from typing import List, Dict, Any
from pathlib import Path

from PIL import Image
from pydantic.color import Color

from terra_ai.cascades.common import decamelize
from terra_ai.cascades.create import json2cascade
from terra_ai.data.cascades.blocks.extra import BlockFunctionGroupChoice, FunctionParamsChoice, \
    ObjectDetectionFilterClassesList, BlockServiceTypeChoice
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.datasets.extra import LayerInputTypeChoice
from terra_ai.data.deploy.extra import DeployTypeChoice
from terra_ai.deploy.create_deploy_package import CascadeCreator
from terra_ai.settings import DEPLOY_PATH
from terra_ai.utils import camelize


class CascadeRunner:

    def start_cascade(self, cascade_data: CascadeDetailsData, training_path: Path,
                      sources: Dict[int, List[str]]):
        # print('sources', sources)
        config = CascadeCreator()

        presets_path = os.path.join(DEPLOY_PATH, "deploy_presets")
        if os.path.exists(DEPLOY_PATH):
            shutil.rmtree(DEPLOY_PATH, ignore_errors=True)
            os.makedirs(DEPLOY_PATH, exist_ok=True)
        if not os.path.exists(presets_path):
            os.makedirs(presets_path, exist_ok=True)

        type_, model, inputs_ids = self._get_task_type(cascade_data=cascade_data, training_path=training_path)
        # print('type_', type_)
        if model:
            dataset_path = os.path.join(training_path, model, "model", "dataset.json")
            if not os.path.exists(dataset_path):
                dataset_path = os.path.join(training_path, model, "model", "dataset", "config.json")
            with open(dataset_path, "r", encoding="utf-8") as dataset_config:
                dataset_config_data = json.load(dataset_config)
            model_path = Path(os.path.join(training_path, model, "model"))
            cascade_path = os.path.join(training_path, model)
            config.copy_model(deploy_path=DEPLOY_PATH, model_path=model_path)
            model_task = list(set([val.get("task") for key, val in dataset_config_data.get("outputs").items()]))[0]
        else:
            dataset_config_data = None
            cascade_path = None
            model_task = type_

        cascade_config, classes, classes_colors = self._create_config(cascade_data=cascade_data,
                                                                      model_task=model_task,
                                                                      dataset_data=dataset_config_data,
                                                                      presets_path=presets_path)
        # print('cascade_config', cascade_config)
        main_block = json2cascade(path=cascade_path, cascade_config=cascade_config, mode="run")

        sources = sources.get(inputs_ids[0])

        presets_data = self._get_presets(sources=sources, type_=type_, cascade=main_block,
                                         predict_path=str(DEPLOY_PATH), classes=classes,
                                         classes_colors=classes_colors)
        out_data = dict([
            ("path_deploy", str(DEPLOY_PATH)),
            ("type", type_),
            ("data", presets_data)
        ])

        with open(os.path.join(presets_path, "presets_config.json"), "w", encoding="utf-8") as config:
            json.dump(out_data, config)

        return cascade_config

    @staticmethod
    def _get_task_type(cascade_data: CascadeDetailsData, training_path: Path):
        model = "__current"
        _inputs = []
        _input_type = "video"
        deploy_type = DeployTypeChoice.VideoObjectDetection
        for block in cascade_data.blocks:
            if block.group == BlockGroupChoice.Model:
                model = block.parameters.main.path
            if block.group == BlockGroupChoice.InputData:
                _inputs.append(block.id)
                if block.parameters.main.type == LayerInputTypeChoice.Image:
                    _input_type = "image"
            if block.parameters.main.type in [BlockServiceTypeChoice.YoloV5,
                                              BlockServiceTypeChoice.GoogleTTS,
                                              BlockServiceTypeChoice.TinkoffAPI,
                                              BlockServiceTypeChoice.Wav2Vec]:
                model = None
                if block.parameters.main.type == BlockServiceTypeChoice.YoloV5 and _input_type == "image":
                    deploy_type = DeployTypeChoice.YoloV5
                if block.parameters.main.type == BlockServiceTypeChoice.Wav2Vec:
                    deploy_type = DeployTypeChoice.Wav2Vec
                    _input_type = "audio"
                if block.parameters.main.type == BlockServiceTypeChoice.GoogleTTS:
                    deploy_type = DeployTypeChoice.GoogleTTS
                if block.parameters.main.type == BlockServiceTypeChoice.TinkoffAPI:
                    deploy_type = DeployTypeChoice.TinkoffAPI

        if model:
            with open(os.path.join(training_path, model, "config.json"),
                      "r", encoding="utf-8") as training_config:
                training_details = json.load(training_config)
            if deploy_type not in [DeployTypeChoice.YoloV3, DeployTypeChoice.YoloV4]:
                deploy_type = DeployTypeChoice(training_details.get("base").get("architecture").get("type"))

        return deploy_type, model, _inputs

    def _create_config(self, cascade_data: CascadeDetailsData, model_task: str,
                       dataset_data: dict, presets_path: str):
        if dataset_data:
            classes = list(dataset_data.get("outputs").values())[0].get("classes_names")
            num_class = list(dataset_data.get("outputs").values())[0].get("num_classes")
            classes_colors = []
            if list(dataset_data.get("outputs").values())[0].get("classes_colors"):
                classes_colors = [list(Color(color).as_rgb_tuple()) for color in
                                  list(dataset_data.get("outputs").values())[0].get("classes_colors")]
        else:
            classes = ObjectDetectionFilterClassesList
            classes_colors = None

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
                print('model_task', model_task)
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
                elif model_task in [DeployTypeChoice.GoogleTTS]:
                    block_description = {
                        "saving": {
                            "tag": "output",
                            "type": "audio"
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
                    if "filter_classes" in parameters.keys():
                        parameters["filter_classes"] = [ObjectDetectionFilterClassesList.index(x)
                                                        for x in parameters["filter_classes"]]
                    # if "num_class" in parameters.keys() and not parameters.get("num_class"):
                    #     parameters["num_class"] = num_class
                    if "classes_colors" in parameters.keys() and not parameters.get("classes_colors"):
                        parameters["classes_colors"] = classes_colors
                elif block.group == BlockGroupChoice.Service:
                    _tag = block.group.value.lower()
                    _task = decamelize(block.parameters.main.group.value)  # .lower()
                    _name = block.parameters.main.type.value
                    block_parameters = FunctionParamsChoice.get_parameters(input_block=block.parameters.main.type)
                    parameters = {
                        key: val for key, val in block.parameters.main.native().items()
                        if key in block_parameters
                    }
                    if "model_path" in parameters.keys() and not parameters.get("model_path"):
                        parameters["model_path"] = "terra_ai/assets/cascades/ckpt.t7"
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

        with open(os.path.join(presets_path, "cascade_config.json"), "w", encoding="utf-8") as cascade_config:
            json.dump(config, cascade_config)

        return config, classes, classes_colors

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

    def _get_presets(self, sources: List[Any], type_: DeployTypeChoice, cascade: Any,
                     predict_path: str, classes: list, classes_colors: list):

        out_data = []
        iter_ = 0
        for source in sources[:10]:
            if type_ in [DeployTypeChoice.YoloV3, DeployTypeChoice.YoloV4,
                         DeployTypeChoice.YoloV5, DeployTypeChoice.VideoObjectDetection]:
                if type_ == DeployTypeChoice.VideoObjectDetection:
                    data_type = "video"
                    predict_file_name = f"deploy_presets/result_{iter_}.webm"
                    source_file_name = f"deploy_presets/initial_{iter_}.webm"
                else:
                    data_type = "image"
                    predict_file_name = f"deploy_presets/result_{iter_}.webp"
                    source_file_name = f"deploy_presets/initial_{iter_}.webp"

                output_path = os.path.join(predict_path, predict_file_name)
                self._save_web_format(initial_path=source,
                                      deploy_path=os.path.join(predict_path, source_file_name),
                                      source_type=data_type)
                cascade(input_path=source, output_path=output_path)
                out_data.append({
                    "source": source_file_name,
                    "predict": predict_file_name
                })
            elif type_ == DeployTypeChoice.GoogleTTS:
                predict_file_name = f"deploy_presets/result_{iter_}.webm"
                source_file_name = f"deploy_presets/initial_{iter_}.txt"

                input_path = os.path.join(predict_path, source_file_name)
                output_path = os.path.join(predict_path, predict_file_name)

                with open(input_path, "w", encoding="utf-8") as source_file:
                    source_file.write(source)

                cascade(input_path=input_path, output_path=output_path)
                out_data.append({
                    "source": source_file_name,
                    "predict": predict_file_name
                })
            elif type_ in [DeployTypeChoice.Wav2Vec, DeployTypeChoice.TinkoffAPI]:
                data_type = "audio"
                predict_file_name = f"deploy_presets/result_{iter_}.txt"
                source_file_name = f"deploy_presets/initial_{iter_}.webm"

                output_path = os.path.join(predict_path, predict_file_name)
                self._save_web_format(initial_path=source,
                                      deploy_path=os.path.join(predict_path, source_file_name),
                                      source_type=data_type)
                cascade(input_path=source, output_path=output_path)

                with open(output_path, "w", encoding="utf-8") as out_file:
                    out_file.write(cascade.out)

                out_data.append({
                    "source": source_file_name,
                    "predict": predict_file_name
                })
            elif type_ in [DeployTypeChoice.ImageClassification, DeployTypeChoice.AudioClassification,
                           DeployTypeChoice.TextClassification, DeployTypeChoice.VideoClassification,
                           DeployTypeChoice.DataframeClassification]:
                # if type_ == DeployTypeChoice.ImageClassification:
                #     data_type = "image"
                #     source_file_name = f"deploy_presets/initial_{iter_}.webp"
                # elif type_ == DeployTypeChoice.AudioClassification:
                #     data_type = "audio"
                #     source_file_name = f"deploy_presets/initial_{iter_}.webm"
                # elif type_ == DeployTypeChoice.VideoClassification:
                #     data_type = "video"
                #     source_file_name = f"deploy_presets/initial_{iter_}.webm"
                # self._save_web_format(initial_path=input_path,
                #                       deploy_path=os.path.join(predict_path, source_file_name),
                #                       source_type=data_type)
                # cascade(input_path=input_path)
                # print(source)
                # out_data.append({
                #     "source": source_file_name,
                #     "actual": cascade.out,
                #     "data": cascade.out
                # })
                # print(out_data)
                pass
            elif type_ in [DeployTypeChoice.ImageSegmentation]:
                data_type = "image"
                predict_file_name = f"deploy_presets/result_{iter_}.webp"
                source_file_name = f"deploy_presets/initial_{iter_}.webp"
                output_path = os.path.join(predict_path, predict_file_name)

                self._save_web_format(initial_path=source,
                                      deploy_path=os.path.join(predict_path, source_file_name),
                                      source_type=data_type)
                cascade(input_path=source, output_path=output_path)
                mask = cascade[0][1].out
                sum_list = [np.sum(mask[:, :, :, i]) for i in range(mask.shape[-1])]
                example_data = [(classes[i], classes_colors[i]) for i, count in enumerate(sum_list) if count > 0]
                out_data.append({
                    "source": source_file_name,
                    "segment": predict_file_name,
                    "data": example_data
                })
            iter_ += 1

        return {"data": out_data}

    @staticmethod
    def _save_web_format(initial_path: str, deploy_path: str, source_type: str):
        if source_type == "image":
            img = Image.open(initial_path)
            img.save(deploy_path, 'webp')
        elif source_type == "video":
            clip = moviepy_editor.VideoFileClip(initial_path)
            clip.write_videofile(deploy_path)
        elif source_type == "audio":
            AudioSegment.from_file(initial_path).export(deploy_path, format="webm")
