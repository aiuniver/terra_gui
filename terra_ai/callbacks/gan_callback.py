import os
from collections import Counter
from random import shuffle

import matplotlib
import numpy as np
import tensorflow
from PIL import Image
from tensorflow.keras.preprocessing import image

from terra_ai.callbacks.utils import get_y_true, round_loss_metric, sort_dict, fill_graph_front_structure, \
    fill_graph_plot_data, get_link_from_dataframe
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice
import terra_ai.exceptions.callbacks as exception
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.logging import logger
from terra_ai.settings import DEPLOY_PRESET_PERCENT


class GANCallback:
    name = 'GANCallback'

    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        # logger.debug(f"{GANCallback.name}, {GANCallback.get_x_array.__name__}")
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        # logger.debug(f"{GANCallback.name}, {GANCallback.postprocess_deploy.__name__}")
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
                    seed_array=None,
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
                               y_pred, inverse_y_pred, raw_y_pred=None) -> dict:
        return {}

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config) -> dict:
        return {}

    @staticmethod
    def prepare_example_idx_to_show(array: np.ndarray, seed_array: np.ndarray, count: int,
                                    choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.seed) -> np.ndarray:
        # logger.debug(f"{GANCallback.name}, {GANCallback.prepare_example_idx_to_show.__name__}")
        method_name = 'prepare_example_idx_to_show'
        try:
            if choice_type == ExampleChoiceTypeChoice.seed:
                example_idx = np.array(seed_array[:count*5], dtype='float32')
            else:
                example_idx = np.array(array[:count * 5], dtype='float32')
                # example_idx = np.array(array[np.random.randint(0, len(array), count*5)], dtype='float32')
            return example_idx
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_gan(predict_array: np.ndarray, image_id: int, save_path, return_mode='deploy'):
        # logger.debug(f"{GANCallback.name}, {GANCallback.postprocess_gan.__name__}")
        method_name = 'postprocess_gan'
        try:
            # data = {"y_true": {}, "y_pred": {}, "stat": {}}
            if return_mode == 'deploy':
                # predict_array = predict_array.astype("uint8")
                img_save_path = os.path.join(save_path, "deploy_presets", f"gan_postprocessing_{image_id}.webp")
                return_path = os.path.join("deploy_presets", f"gan_postprocessing_{image_id}.webp")
                matplotlib.image.imsave(img_save_path, predict_array)
                return return_path

            if return_mode == 'callback':
                data = {"type": "image", "data": []}
                _id = 1
                for array in predict_array:
                    array = array / array.max()
                    # print(array.shape, type(array))
                    y_pred_save_path = os.path.join(save_path, f"predict_gan_image_{image_id}_position_{_id}.webp")
                    matplotlib.image.imsave(y_pred_save_path, array)
                    data["data"].append(
                        {
                            "title": "Изображение",
                            "value": y_pred_save_path,
                            "color_mark": None,
                            # "size": "large"
                        }
                    )
                    _id += 1
                return data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_initial_source(options: PrepareDataset, image_id: int, preset_path: str):
        method_name = 'postprocess_initial_source'
        try:
            total_length = len(options.dataframe.get('train'))
            idxs = np.random.randint(0, total_length, 3)
            column_disc = ''
            for column in options.dataframe.get('train').columns:
                if 'Изображения' in column and 'sources' in options.dataframe.get('train')[column][0]:
                    column_disc = column
                    break
            _id = 1
            data = {"type": "image", "data": []}
            for idx in idxs:
                initial_file_path = get_link_from_dataframe(
                    dataframe=options.dataframe.get('train'),
                    column=column_disc,
                    index=idx
                )
                img = Image.open(os.path.join(options.data.path, initial_file_path))
                img = img.resize(
                    options.data.inputs.get(int(column_disc.split('_')[0])).shape[0:2][::-1],
                    Image.ANTIALIAS
                )
                img = img.convert('RGB')
                source = os.path.join(preset_path, f"initial_gan_image_{image_id}_position_{_id}.webp")
                img.save(source, 'webp')
                data["data"].append(
                    {
                        "title": "Изображение",
                        "value": source,
                        "color_mark": None,
                    }
                )
                _id += 1
            return data

        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors, raw_y_pred=None):
        # logger.debug(f"{GANCallback.name}, {GANCallback.intermediate_result_request.__name__}")
        method_name = 'intermediate_result_request'
        try:
            return_data = {}
            if interactive_config.intermediate_result.show_results:
                # data_type = interactive_config.intermediate_result.data_type.name
                for idx in range(interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': None,
                        'statistic_values': {}
                    }
                    init_data = GANCallback().postprocess_initial_source(
                        options=options,
                        image_id=idx,
                        preset_path=preset_path,
                    )
                    return_data[f"{idx + 1}"]['initial_data'][f"Тренировочные данные"] = init_data
                    pred_data = GANCallback().postprocess_gan(
                        predict_array=example_idx[idx*3: (idx+1)*3],
                        image_id=idx,
                        save_path=preset_path,
                        return_mode='callback'
                    )
                    return_data[f"{idx + 1}"]['predict_value'][f"Генератор"] = pred_data
                return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc


class CGANCallback:
    name = 'CGANCallback'

    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        return None, None

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        # TODO: актуализировать когда появятся темплейты для деплоя
        # logger.debug(f"{CGANCallback.name}, {CGANCallback.postprocess_deploy.__name__}")
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
                    seed_array=None,
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
                CGANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def dataset_balance(options: PrepareDataset, y_true, preset_path: str, class_colors) -> dict:
        class_list = []
        for col in options.dataframe['train'].columns:
            if 'Класс' in col:
                class_list = list(options.dataframe['train'][col])
                break
        return dict(Counter(class_list))

    @staticmethod
    def statistic_data_request(interactive_config, options, y_true, inverse_y_true,
                               y_pred, inverse_y_pred, raw_y_pred=None) -> dict:
        return {}


    @staticmethod
    def prepare_example_idx_to_show(array: dict, seed_array: dict, count: int,
                                    choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.seed) -> dict:
        # logger.debug(f"{CGANCallback.name}, {CGANCallback.prepare_example_idx_to_show.__name__}")
        method_name = 'prepare_example_idx_to_show'
        try:
            example_idx = {}
            if choice_type == ExampleChoiceTypeChoice.seed:
                name_list = list(seed_array.keys())
                shuffle_idx = list(np.arange(len(name_list)))
                shuffle(shuffle_idx)
                for i in shuffle_idx:
                    example_idx[name_list[i]] = np.array(seed_array[name_list[i]][:3], dtype='float32')
                    # if i == count:
                    #     break
            else:
                name_list = list(array.keys())
                shuffle(name_list)
                for i, name in enumerate(name_list):
                    examples = np.random.choice(np.arange(len(array[name])), 3)
                    example_idx[name] = np.array(array[name][examples], dtype='float32')
                    # if i == count:
                    #     break
            return example_idx
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                CGANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_gan(predict_array: np.ndarray, label: str, image_id: int, save_path, return_mode='deploy'):
        # logger.debug(f"{CGANCallback.name}, {CGANCallback.postprocess_gan.__name__}")
        method_name = 'postprocess_gan'
        try:
            # data = {"y_true": {}, "y_pred": {}, "stat": {}}
            if return_mode == 'deploy':
                # predict_array = predict_array.astype("uint8")
                img_save_path = os.path.join(save_path, "deploy_presets", f"gan_postprocessing_{image_id}.webp")
                return_path = os.path.join("deploy_presets", f"gan_postprocessing_{image_id}.webp")
                matplotlib.image.imsave(img_save_path, predict_array)
                return return_path

            if return_mode == 'callback':
                _id = 1
                # data = {
                #     "type": "str",
                #     "data": [
                #         {
                #             "title": "Класс",
                #             "value": label,
                #             "color_mark": None
                #         }
                #     ]
                # }
                data = {"type": "image", "data": []}
                for array in predict_array:
                    array = array / array.max()
                    # print(array.shape, type(array))
                    y_pred_save_path = os.path.join(save_path, f"predict_gan_image_{image_id}_position_{_id}.webp")
                    matplotlib.image.imsave(y_pred_save_path, array)
                    data["data"].append(
                        {
                            "title": "Изображение",
                            "value": y_pred_save_path,
                            "color_mark": None,
                            # "size": "large"
                        }
                    )
                    _id += 1
                return data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                CGANCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_initial_source(options: PrepareDataset, label, labels: dict, image_id: int, preset_path: str):
        method_name = 'postprocess_initial_source'
        try:
            # total_length = len(labels.get(label))
            idxs = np.random.choice(labels.get(label), 3)
            column_disc = ''
            for column in options.dataframe.get('train').columns:
                if 'Изображения' in column and 'sources' in options.dataframe.get('train')[column][0]:
                    column_disc = column
                    break
            _id = 1
            data = {"type": "image", "data": []}
            for idx in idxs:
                # logger.debug(f"label: {label} - {options.dataframe.get('train')['2_Класс'][idx]}")
                initial_file_path = get_link_from_dataframe(
                    dataframe=options.dataframe.get('train'),
                    column=column_disc,
                    index=idx
                )
                img = Image.open(os.path.join(options.data.path, initial_file_path))
                img = img.resize(
                    options.data.inputs.get(int(column_disc.split('_')[0])).shape[0:2][::-1],
                    Image.ANTIALIAS
                )
                img = img.convert('RGB')
                source = os.path.join(preset_path, f"initial_gan_image_{image_id}_position_{_id}.webp")
                img.save(source, 'webp')
                data["data"].append(
                    {
                        "title": "Изображение",
                        "value": source,
                        "color_mark": None,
                    }
                )
                _id += 1
            return data

        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                CGANCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_label_idx(options: PrepareDataset):
        labels = []
        for col in options.dataframe.get('train').columns:
            if "Класс" in col:
                labels = list(options.dataframe['train'][col])
        label_idx = {}
        for i, lbl in enumerate(labels):
            if f"{lbl}" not in label_idx.keys():
                label_idx[f"{lbl}"] = []
            label_idx[f"{lbl}"].append(i)
        # for i in label_idx.keys():
        #     logger.debug(f"label {i}: {label_idx[i][:10]}")
        return label_idx


    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors, raw_y_pred=None):
        # logger.debug(f"{CGANCallback.name}, {CGANCallback.intermediate_result_request.__name__}")
        method_name = 'intermediate_result_request'
        try:
            return_data = {}
            label_idx = CGANCallback().get_label_idx(options)
            # logger.debug(f"label_idx: {label_idx.keys()}")
            if interactive_config.intermediate_result.show_results:
                # data_type = interactive_config.intermediate_result.data_type.name
                for idx in range(interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': None,
                        'statistic_values': {}
                    }
                    label = list(example_idx.keys())[idx]
                    # logger.debug(f"label: {label}")
                    return_data[f"{idx + 1}"]['initial_data'][f"Класс"] = {
                        "type": "str",
                        "data": [
                            {
                                "title": "Класс",
                                "value": label,
                                "color_mark": None
                            }
                        ]
                    }
                    true_data = CGANCallback().postprocess_initial_source(
                        options=options,
                        label=label,
                        labels=label_idx,
                        image_id=idx,
                        preset_path=preset_path,
                    )
                    return_data[f"{idx + 1}"]['true_value'][f"Тренировочные данные"] = true_data

                    pred_data = CGANCallback().postprocess_gan(
                        predict_array=example_idx[label],
                        label=label,
                        image_id=idx,
                        save_path=preset_path,
                        return_mode='callback'
                    )
                    return_data[f"{idx + 1}"]['predict_value'][f"Генератор"] = pred_data

                return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                CGANCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config):
        method_name = 'balance_data_request'
        try:
            return_data = []
            _id = 0
            preset = {}
            class_names, class_count = sort_dict(
                dict_to_sort=dataset_balance,
                mode=interactive_config.data_balance.sorted.name
            )
            preset['train'] = fill_graph_front_structure(
                _id=0,
                _type='histogram',
                type_data='train',
                graph_name=f"Тренировочная выборка",
                short_name=f"Тренировочная",
                x_label="Название класса",
                y_label="Значение",
                plot_data=[fill_graph_plot_data(x=class_names, y=class_count)],
            )
            return_data.append(preset)
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                CGANCallback().name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

