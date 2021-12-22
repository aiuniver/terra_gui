import os

import numpy as np
import tensorflow
from PIL import Image
from tensorflow.keras.preprocessing import image

from terra_ai.callbacks.utils import get_y_true, round_loss_metric
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice
import terra_ai.exceptions.callbacks as exception
from terra_ai.logging import logger
from terra_ai.settings import DEPLOY_PRESET_PERCENT


class GANCallback:
    name = 'GANCallback'

    def __init__(self):
        self.noise = 100

    @staticmethod
    def get_x_array(options):
        x_val = {}
        inverse_x_val = {}
        return x_val, inverse_x_val

    @staticmethod
    def get_y_true(options, dataset_path=""):
        y_true = {"train": {}, "val": {}}
        inverse_y_true = {"train": {}, "val": {}}
        return y_true, inverse_y_true

    @staticmethod
    def get_inverse_array(array: dict, options, type="output"):
        inverse_array = {"train": {}, "val": {}}
        return inverse_array

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        method_name = 'postprocess_deploy'
        try:
            return_data = {}
            for i, output_id in enumerate(options.data.outputs.keys()):
                true_array = get_y_true(options, output_id)
                if len(options.data.outputs.keys()) > 1:
                    postprocess_array = array[i]
                else:
                    postprocess_array = array
                example_idx = GANCallback().prepare_example_idx_to_show(
                    array=postprocess_array[:len(array)],
                    true_array=true_array[:len(array)],
                    count=int(len(array) * DEPLOY_PRESET_PERCENT / 100)
                )
                return_data[output_id] = {'preset': [], 'label': []}
                source_col = []
                for inp in options.data.inputs.keys():
                    source_col.extend(list(options.data.columns.get(inp).keys()))
                preprocess = options.preprocessing.preprocessing.get(output_id)
                for idx in example_idx:
                    row_list = []
                    for inp_col in source_col:
                        row_list.append(f"{options.dataframe.get('val')[inp_col][idx]}")
                    return_data[output_id]['preset'].append(row_list)
                    channel_inverse_col = []
                    for ch, col in enumerate(list(options.data.columns.get(output_id).keys())):
                        channel_inverse_col = []
                        if type(preprocess.get(col)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {int(output_id): {col: array[idx, ch:ch + 1].reshape(-1, 1)}}
                            inverse_col = options.preprocessing.inverse_data(_options).get(output_id).get(col)
                            inverse_col = inverse_col.squeeze().astype('float').tolist()
                        else:
                            inverse_col = array[idx, ch:ch + 1].astype('float').tolist()
                        channel_inverse_col.append(round_loss_metric(inverse_col))
                    return_data[output_id]['label'].append(channel_inverse_col)  # [0]
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def dataset_balance(options, y_true, preset_path: str, class_colors) -> dict:
        return {}

    @staticmethod
    def statistic_data_request(interactive_config, options, y_true, inverse_y_true,
                               y_pred, inverse_y_pred, raw_y_pred=None) -> list:
        return []

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config):
        return []

    @staticmethod
    def prepare_example_idx_to_show(self, array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                    choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.best,
                                    seed_idx: dict = None, noise: int = None) -> dict:
        method_name = 'prepare_example_idx_to_show'
        try:
            if choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
                example_idx = seed_idx['train'][:count]
            else:
                example_idx = {'train': tensorflow.random.normal(shape=(count, 5, noise))}
            return example_idx
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_gan(column_names: list, inverse_y_true: np.ndarray, inverse_y_pred: np.ndarray,
                        show_stat: bool = False, return_mode='deploy'):
        method_name = 'postprocess_regression'
        try:
            data = {"y_true": {"type": "str", "data": []}}
            if return_mode == 'deploy':
                source = []
                return source
            else:
                for i, name in enumerate(column_names):
                    data["y_true"]["data"].append(
                        {"title": name.split('_', 1)[-1], "value": f"{inverse_y_true[i]: .2f}", "color_mark": None}
                    )
                deviation = np.abs((inverse_y_pred - inverse_y_true) * 100 / inverse_y_true)
                data["y_pred"] = {
                    "type": "str",
                    "data": []
                }
                for i, name in enumerate(column_names):
                    color_mark = 'success' if deviation[i] < 2 else "wrong"
                    data["y_pred"]["data"].append(
                        {
                            "title": name.split('_', 1)[-1],
                            "value": f"{inverse_y_pred[i]: .2f}",
                            "color_mark": color_mark
                        }
                    )
                if show_stat:
                    data["stat"] = {"type": "str", "data": []}
                    for i, name in enumerate(column_names):
                        color_mark = 'success' if deviation[i] < 2 else "wrong"
                        data["stat"]["data"].append(
                            {
                                'title': f"Отклонение - «{name.split('_', 1)[-1]}»",
                                'value': f"{np.round(deviation[i], 2)} %",
                                'color_mark': color_mark
                            }
                        )
                return data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc



    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, dataset_path: str, preset_path: str,
                                   data_type: str = 'val', save_id: int = None, x_array=None, return_mode='deploy'):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            if options.data.group != DatasetGroupChoice.keras:
                for inp in options.data.inputs.keys():
                    for column_name in options.dataframe.get(data_type).columns:
                        if column_name.split('_')[0] == f"{inp}":
                            column_idx.append(options.dataframe.get(data_type).columns.tolist().index(column_name))
                initial_file_path = os.path.join(
                    dataset_path, options.dataframe.get(data_type).iat[example_id, column_idx[0]]
                )
                if not save_id:
                    return str(os.path.abspath(initial_file_path))
            else:
                initial_file_path = ""

            if options.data.group != DatasetGroupChoice.keras:
                img = Image.open(initial_file_path)
                img = img.resize(
                    options.data.inputs.get(input_id).shape[0:2][::-1],
                    Image.ANTIALIAS
                )
            else:
                if x_array is None:
                    x_array = options.X.get(data_type).get(f"{list(options.data.inputs.keys())[0]}")
                img = image.array_to_img(x_array[example_id])
            img = img.convert('RGB')

            if return_mode == 'deploy':
                source = os.path.join(preset_path, "deploy_presets",
                                      f"initial_data_image_{save_id}_input_{input_id}.webp")
                return_source = os.path.join("deploy_presets", f"initial_data_image_{save_id}_input_{input_id}.webp")
                img.save(source, 'webp')
                return return_source
            if return_mode == 'callback':
                source = os.path.join(preset_path, f"initial_data_image_{save_id}_input_{input_id}.webp")
                img.save(source, 'webp')
                data = [
                    {
                        "title": "Изображение",
                        "value": source,
                        "color_mark": None
                    }
                ]
                return data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ImageClassificationCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        method_name = 'postprocess_deploy'
        try:
            x_array, inverse_x_array = GANCallback.get_x_array(options)
            return_data = {}
            if array is None:
                logger.warning("postprocess_deploy: array is None")

            for i, output_id in enumerate(options.data.outputs.keys()):
                true_array = get_y_true(options, output_id)
                if len(options.data.outputs.keys()) > 1:
                    postprocess_array = array[i]
                else:
                    postprocess_array = array
                example_idx = GANCallback.prepare_example_idx_to_show(
                    array=postprocess_array[:len(array)],
                    true_array=true_array[:len(array)],
                    options=options,
                    output=output_id,
                    count=int(len(array) * DEPLOY_PRESET_PERCENT / 100)
                )

                return_data[output_id] = []
                _id = 1
                for idx in example_idx:
                    input_id = list(options.data.inputs.keys())[0]
                    source = GANCallback.postprocess_initial_source(
                        options=options,
                        input_id=input_id,
                        save_id=_id,
                        example_id=idx,
                        dataset_path=dataset_path,
                        preset_path=save_path,
                        x_array=None if not x_array else x_array.get(f"{input_id}"),
                        return_mode='deploy'
                    )
                    actual_value, predict_values = GANCallback.postprocess_classification(
                        predict_array=np.expand_dims(postprocess_array[idx], axis=0),
                        true_array=true_array[idx],
                        options=options.data.outputs[output_id],
                        return_mode='deploy'
                    )
                    return_data[output_id].append(
                        {
                            "source": source,
                            "actual": actual_value,
                            "data": predict_values[0]
                        }
                    )
                    _id += 1

            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors, raw_y_pred=None):
        method_name = 'intermediate_result_request'
        try:
            return_data = {}
            if interactive_config.intermediate_result.show_results:
                data_type = interactive_config.intermediate_result.data_type.name
                for idx in range(interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': None,
                        'statistic_values': {}
                    }
                    for inp in options.data.inputs.keys():
                        data = GANCallback.postprocess_initial_source(
                            options=options,
                            input_id=inp,
                            save_id=idx + 1,
                            example_id=example_idx[idx],
                            dataset_path=dataset_path,
                            preset_path=preset_path,
                            x_array=x_val.get(data_type).get(f"{inp}") if x_val else None,
                            return_mode='callback',
                            data_type=data_type
                        )
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'type': 'image',
                            'data': data,
                        }

                    for out in options.data.outputs.keys():
                        data = GANCallback().postprocess_classification(
                            predict_array=y_pred.get(data_type).get(f'{out}')[example_idx[idx]],
                            true_array=y_true.get(data_type).get(f'{out}')[example_idx[idx]],
                            options=options.data.outputs.get(out),
                            show_stat=interactive_config.intermediate_result.show_statistic,
                            return_mode='callback'
                        )
                        if data.get('y_true'):
                            return_data[f"{idx + 1}"]['true_value'][f"Выходной слой «{out}»"] = data.get('y_true')
                        return_data[f"{idx + 1}"]['predict_value'][f"Выходной слой «{out}»"] = data.get('y_pred')
                        return_data[f"{idx + 1}"]['tags_color'] = None
                        if data.get('stat'):
                            return_data[f"{idx + 1}"]['statistic_values'][f"Выходной слой «{out}»"] = data.get('stat')
                        else:
                            return_data[f"{idx + 1}"]['statistic_values'] = {}
                return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc
