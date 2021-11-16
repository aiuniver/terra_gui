import json
import os

import pandas as pd
from pathlib import Path

from terra_ai.cascades.common import decamelize
from terra_ai.cascades.create import json2cascade
from terra_ai.data.cascades.blocks.extra import BlockFunctionGroupChoice
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice


class CascadeRunner:

    def start_cascade(self, cascade_data: CascadeDetailsData, path: Path):
        dataset_path = os.path.join(os.path.split(path)[0], "datasets")

        with open(os.path.join(dataset_path, "config.json"), "r", encoding="utf-8") as dataset_config:
            dataset_config_data = json.load(dataset_config)

        model_task = list(set([val.get("task") for key, val in dataset_config_data.get("outputs").items()]))[0]
        cascade_config = self._create_config(cascade_data=cascade_data, model_task=model_task)
        script_name, model = self._get_task_type(cascade_data=cascade_data, training_path=path)

        main_block = json2cascade(path=os.path.join(path, model), cascade_config=cascade_config, mode="run")
        sources = self._get_sources(dataset_path=dataset_path)
        i = 0
        print(sources)
        for source in sources:
            if "text" in script_name:
                with open("test.txt", "w", encoding="utf-8") as f:
                    f.write(source)
                input_path = "test.txt"
            else:
                input_path = os.path.join(dataset_path, source)
            if "segmentation" in script_name:
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

    def _create_config(self, cascade_data: CascadeDetailsData, model_task: str):
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
                block_description = {
                    block.name: {
                        "tag": block.parameters.main.type.value.lower(),
                        "task": block.parameters.main.group.value,
                        "name": decamelize(block.parameters.main.type.value),
                        "params": {
                            key: val for key, val in block.parameters.main.native().items()
                            if key not in ["type", "group"]
                        }
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
        for block in cascade_data.blocks:
            if block.id in blocks_ids:
                if block.group == BlockGroupChoice.InputData:
                    mapping.append("INPUT")
                elif block.group == BlockGroupChoice.Model:
                    mapping.append("model")
                else:
                    mapping.append(block.name)
        return mapping

    @staticmethod
    def _get_sources(dataset_path: str) -> list:
        out = []
        sources = pd.read_csv(os.path.join(dataset_path, "instructions", "tables", "val.csv"))
        print(sources)
        for column in sources.columns:
            if column.split("_")[-1].title() in ["Image", "Text", "Audio", "Video"]:
                out = sources[column].to_list()
        return out
