import json
import os
from pathlib import Path

from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.exceptions import cascades as exceptions


class CascadeValidator:

    def get_validate(self, cascade_data: CascadeDetailsData, training_path: Path):
        configs = self._load_configs(cascade_data=cascade_data, training_path=training_path)
        datasets = configs.get("datasets_config_data")
        models = configs.get("models_config_data")
        model_data_type = list(set([val.get("task") for key, val in models[0].get("inputs").items()]))[0]
        result = self._check_bind_and_data(cascade_data=cascade_data, model_data_type=model_data_type)
        return result

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

        for path in models_paths:
            with open(os.path.join(training_path, path, "model", "dataset", "config.json"),
                      "r", encoding="utf-8") as model_config:
                model_config_data = json.load(model_config)
            models.append(model_config_data)
        return {
            "datasets_config_data": datasets,
            "models_config_data": models
        }

    def _check_bind_and_data(self, cascade_data: CascadeDetailsData, model_data_type: str):
        bind_errors = dict()
        blocks = cascade_data.blocks
        for block in blocks:
            if block.group == BlockGroupChoice.InputData:
                if block.bind.up or not block.bind.down:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(exceptions.BlockNotConnectedToMainPartException()))
                # if block.parameters.main.type != dataset_data_type:
                #     bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                #                                   error=str(exceptions.DatasetDataDoesNotMatchInputDataException(
                #                                       dataset_data_type, block.parameters.main.type
                #                                   )))
                if block.parameters.main.type != model_data_type:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(exceptions.InputDataDoesNotMatchModelDataException(
                                                      block.parameters.main.type.value, model_data_type
                                                  )))
            elif block.group == BlockGroupChoice.OutputData:
                if not block.bind.up or block.bind.down:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(exceptions.BlockNotConnectedToMainPartException()))
                if block.parameters.main.type != model_data_type:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(exceptions.UsedDataDoesNotMatchBlockDataException(
                                                      block.parameters.main.type.value, model_data_type
                                                  )))
            else:
                if not block.bind.up or not block.bind.down:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(exceptions.BlockNotConnectedToMainPartException()))
                # if block.parameters.main.type != model_data_type:
                #     bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                #                                   error=str(exceptions.UsedDataDoesNotMatchBlockDataException(
                #                                       model_data_type, block.parameters.main.type
                #                                   )))
            if not bind_errors.get(block.id):
                bind_errors[block.id] = None
        return bind_errors

    @staticmethod
    def _add_error(errors: dict, block_id: int, error: str):
        if errors.get(block_id):
            errors[block_id] = f"{errors[block_id]}, {error}"
        else:
            errors[block_id] = error
        return errors

