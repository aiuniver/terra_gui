import json
import os
from pathlib import Path

from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice


class CascadeValidator:

    def get_validate(self, cascade_data: CascadeDetailsData, training_path: Path):
        configs = self._load_configs(cascade_data=cascade_data, training_path=training_path)
        datasets = configs.get("datasets_config_data")
        models = configs.get("models_config_data")
        print(datasets, models)
        validated_cascade = []
        return validated_cascade

    @staticmethod
    def _load_configs(cascade_data: CascadeDetailsData, training_path: Path) -> dict:
        datasets = []
        models = []
        models_paths = [block.parameters.main.path for block in cascade_data.blocks
                        if block.group == BlockGroupChoice.Model]
        dataset_path = os.path.join(os.path.split(training_path)[0], "datasets")
        with open(os.path.join(dataset_path, "config.json"), "r", encoding="utf-8") as dataset_config:
            dataset_config_data = json.load(dataset_config)
        datasets.append(dataset_config_data)

        # for block in cascade_data.blocks:
        #     if block.group == BlockGroupChoice.Model:
        #         models.append(block.parameters.main.path)
        for path in models_paths:
            with open(os.path.join(training_path, path, "model", "dataset", "config.json"),
                      "r", encoding="utf-8") as model_config:
                model_config_data = json.load(model_config)
            models.append(model_config_data)
        return {
            "datasets_config_data": datasets,
            "models_config_data": models
        }
