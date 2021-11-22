import json
import os
from pathlib import Path

from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.cascades.blocks.extra import BlocksBindChoice, BlockServiceTypeChoice
from terra_ai.data.datasets.extra import LayerInputTypeChoice
from terra_ai.exceptions import cascades as exceptions


class CascadeValidator:

    def get_validate(self, cascade_data: CascadeDetailsData, training_path: Path):
        models = self._load_configs(cascade_data=cascade_data, training_path=training_path)
        if models:
            model_data_type = list(set([val.get("task") for key, val in models[0].get("inputs").items()]))[0]
            result = self._check_bind_and_data(cascade_data=cascade_data, model_data_type=model_data_type)
        else:
            instead_block, model_data_type = self._search_block_instead_of_model(cascade_data=cascade_data)
            if instead_block:
                result = self._check_bind_and_data(cascade_data=cascade_data, model_data_type=model_data_type)
            else:
                result = self._add_error(errors={}, block_id=1,
                                         error=str(exceptions.RequiredBlockMissingException(BlockGroupChoice.Model)))
        print(result)
        return result

    @staticmethod
    def _load_configs(cascade_data: CascadeDetailsData, training_path: Path) -> list:
        models = []
        models_paths = [block.parameters.main.path for block in cascade_data.blocks
                        if block.group == BlockGroupChoice.Model]

        for path in models_paths:
            with open(os.path.join(training_path, path, "model", "dataset.json"),
                      "r", encoding="utf-8") as model_config:
                model_config_data = json.load(model_config)
            models.append(model_config_data)
        return models

    def _check_bind_and_data(self, cascade_data: CascadeDetailsData, model_data_type: str):
        bind_errors = dict()
        blocks = cascade_data.blocks
        named_map = self._create_bind_named_map(cascade_data=cascade_data)
        video_by_frame = False
        for block in blocks:
            if block.group == BlockGroupChoice.InputData:
                if block.bind.up or not block.bind.down:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(exceptions.BlockNotConnectedToMainPartException()))
                if block.parameters.main.switch_on_frame:
                    video_by_frame = True
                # if block.parameters.main.type != dataset_data_type:
                #     bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                #                                   error=str(exceptions.DatasetDataDoesNotMatchInputDataException(
                #                                       dataset_data_type, block.parameters.main.type
                #                                   )))
                if block.parameters.main.type != model_data_type:
                    if block.parameters.main.type == LayerInputTypeChoice.Video and \
                            block.parameters.main.switch_on_frame and model_data_type == LayerInputTypeChoice.Image:
                        pass
                    else:
                        bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                      error=str(exceptions.InputDataDoesNotMatchModelDataException(
                                                          block.parameters.main.type.value, model_data_type
                                                      )))
            elif block.group == BlockGroupChoice.OutputData:
                if not block.bind.up or block.bind.down:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(exceptions.BlockNotConnectedToMainPartException()))
                if block.parameters.main.type != model_data_type:
                    if block.parameters.main.type == LayerInputTypeChoice.Video and \
                            not video_by_frame and model_data_type == LayerInputTypeChoice.Image:
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
            if block.group != BlockGroupChoice.InputData:
                input_block, checked_block = self._get_comparison_blocks(block)
                if block.group == BlockGroupChoice.Model:
                    error_args = (checked_block.bind_count,
                                  f'{", ".join(checked_block.binds)}{model_data_type}')
                else:
                    error_args = (checked_block.bind_count, ", ".join(checked_block.binds))

                if len(block.bind.up) < checked_block.bind_count:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(
                                                      exceptions.BindCountNotEnoughException(*error_args)
                                                  ))
                elif len(block.bind.up) > checked_block.bind_count:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(
                                                      exceptions.BindCountExceedingException(*error_args)
                                                  ))
                else:
                    if block.group != BlockGroupChoice.Model:
                        for bind in checked_block.required_binds:
                            checked_binds = [val for key, val in named_map.items() if key in block.bind.up]
                            if bind not in checked_binds:
                                bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                              error=str(exceptions.RequiredBindException(bind)))

                if checked_block.data_type and model_data_type not in checked_block.data_type:
                    bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                  error=str(
                                                      exceptions.BindInappropriateDataTypeException(
                                                          checked_block.data_type,
                                                          model_data_type
                                                      )
                                                  ))

                for block_id in block.bind.up:
                    if checked_block and checked_block != BlocksBindChoice.Model:
                        if checked_block.binds and (named_map.get(block_id) not in checked_block.binds):
                            bind_errors = self._add_error(errors=bind_errors, block_id=block.id,
                                                          error=str(exceptions.ForbiddenBindException(
                                                              named_map.get(block_id),
                                                              ", ".join(checked_block.binds)
                                                          )))

            if not bind_errors.get(block.id):
                bind_errors[block.id] = None
        # for key, val in bind_errors.items():
        #     print(f"{key}: {val}\n")
        return bind_errors

    @staticmethod
    def _add_error(errors: dict, block_id: int, error: str):
        if errors.get(block_id):
            errors[block_id] = f"{errors[block_id]}, {error}"
        else:
            errors[block_id] = error
        return errors

    @staticmethod
    def _create_bind_named_map(cascade_data: CascadeDetailsData):
        mapping = {}
        for block in cascade_data.blocks:
            if block.group in [BlockGroupChoice.Model, BlockGroupChoice.InputData]:
                mapping.update([(block.id, block.group)])
            else:
                mapping.update([(block.id, block.parameters.main.type)])
        return mapping

    @staticmethod
    def _get_comparison_blocks(block):
        if block.group != BlockGroupChoice.InputData:
            input_block = block.group if block.group in [
                BlockGroupChoice.Model,
                BlockGroupChoice.OutputData
            ] else block.parameters.main.type

            checked_block = BlocksBindChoice.checked_block(input_block=input_block)

            return input_block, checked_block
        else:
            return None, None

    @staticmethod
    def _search_block_instead_of_model(cascade_data: CascadeDetailsData):
        for block in cascade_data.blocks:
            if block.parameters.main.type == BlockServiceTypeChoice.YoloV5:
                type_ = BlocksBindChoice.checked_block(input_block=block.parameters.main.type).data_type
                return True, type_
        return False, None
