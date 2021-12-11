import copy
import os

import numpy as np
from PIL import Image
from pydub import AudioSegment
from tensorflow.keras.preprocessing import image

from terra_ai.callbacks.utils import sort_dict, fill_graph_front_structure, fill_graph_plot_data, get_y_true, \
    class_counter, get_confusion_matrix, fill_heatmap_front_structure, get_classification_report, \
    fill_table_front_structure, print_error, round_list
from terra_ai.data.datasets.dataset import DatasetOutputsData
from terra_ai.data.datasets.extra import DatasetGroupChoice, LayerEncodingChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
import moviepy.editor as moviepy_editor

from terra_ai.settings import MAX_GRAPH_LENGTH, DEPLOY_PRESET_PERCENT


class BaseClassificationCallback:
    def __init__(self):
        self.name = 'BaseClassificationCallback'

    @staticmethod
    def get_y_true(options, dataset_path=""):
        method_name = 'get_y_true'
        try:
            y_true = {"train": {}, "val": {}}
            inverse_y_true = {"train": {}, "val": {}}
            for data_type in y_true.keys():
                for out in options.data.outputs.keys():
                    if not options.data.use_generator:
                        y_true[data_type][f"{out}"] = options.Y.get(data_type).get(f"{out}")
                    else:
                        # print('options.dataset', options.dataset)
                        y_true[data_type][f"{out}"] = []
                        for _, y_val in options.dataset[data_type].batch(1):
                            # print('y_val', y_val.keys())
                            y_true[data_type][f"{out}"].extend(y_val.get(f'{out}').numpy())
                        y_true[data_type][f"{out}"] = np.array(y_true[data_type][f"{out}"])
            return y_true, inverse_y_true
        except Exception as e:
            print_error(BaseClassificationCallback().name, method_name, e)

    @staticmethod
    def get_inverse_array(array: dict, options, type="output"):
        inverse_array = {"train": {}, "val": {}}
        # for data_type in inverse_array.keys():
        #     for out in options.data.outputs.keys():
        return inverse_array

    # @staticmethod
    # def get_y_pred(y_true, y_pred, options):
    #     method_name = 'get_y_pred'
    #     try:
    #         reformat_pred = {'train': {}, 'val': {}}
    #         inverse_pred = {'train': {}, 'val': {}}
    #         for data_type in ['train', 'val']:
    #             for idx, out in enumerate(y_true.get('val').keys()):
    #                 if len(y_true.get('val').keys()) == 1:
    #                     reformat_pred[data_type][out] = y_pred
    #                 else:
    #                     reformat_pred[data_type][out] = y_pred[idx]
    #         return reformat_pred, inverse_pred
    #     except Exception as e:
    #         print_error(BaseClassificationCallback().name, method_name, e)

    @staticmethod
    def dataset_balance(options, y_true, preset_path: str, class_colors) -> dict:
        method_name = 'dataset_balance'
        try:
            dataset_balance = {}
            for out in options.data.outputs.keys():
                encoding = options.data.outputs.get(out).encoding
                dataset_balance[f"{out}"] = {'class_histogram': {}}
                for data_type in ['train', 'val']:
                    dataset_balance[f"{out}"]['class_histogram'][data_type] = class_counter(
                        y_array=y_true.get(data_type).get(f"{out}"),
                        classes_names=options.data.outputs.get(out).classes_names,
                        ohe=encoding == LayerEncodingChoice.ohe
                    )
            return dataset_balance
        except Exception as e:
            print_error(BaseClassificationCallback().name, method_name, e)

    @staticmethod
    def statistic_data_request(interactive_config, options, y_true, inverse_y_true,
                               y_pred, inverse_y_pred, raw_y_pred=None) -> list:
        method_name = 'statistic_data_request'
        try:
            print(method_name)
            return_data = []
            _id = 1
            for out in interactive_config.statistic_data.output_id:
                encoding = options.data.outputs.get(out).encoding
                for data_type in y_true.keys():
                    type_name = "Тренировочная" if data_type == 'train' else "Проверочная"
                    if encoding != LayerEncodingChoice.multi:
                        cm, cm_percent = get_confusion_matrix(
                            y_true=np.argmax(y_true.get(data_type).get(f'{out}'), axis=-1)
                            if encoding == LayerEncodingChoice.ohe else y_true.get("val").get(f'{out}'),
                            y_pred=np.argmax(y_pred.get(data_type).get(f'{out}'), axis=-1),
                            get_percent=True
                        )
                        return_data.append(
                            fill_heatmap_front_structure(
                                _id=_id,
                                _type="heatmap",
                                graph_name=f"Выход «{out}» - Confusion matrix - {type_name} выборка",
                                short_name=f"{out} - Confusion matrix - {type_name}",
                                x_label="Предсказание",
                                y_label="Истинное значение",
                                labels=options.data.outputs.get(out).classes_names,
                                data_array=cm,
                                data_percent_array=cm_percent,
                            )
                        )
                        _id += 1

                    else:
                        report = get_classification_report(
                            y_true=y_true.get(data_type).get(f"{out}").reshape(
                                (np.prod(y_true.get(data_type).get(f"{out}").shape[:-1]),
                                 y_true.get(data_type).get(f"{out}").shape[-1])
                            ),
                            y_pred=np.where(y_pred.get(data_type).get(f"{out}") >= 0.9, 1, 0).reshape(
                                (np.prod(y_pred.get(data_type).get(f"{out}").shape[:-1]),
                                 y_pred.get(data_type).get(f"{out}").shape[-1])
                            ),
                            labels=options.data.outputs.get(out).classes_names
                        )
                        return_data.append(
                            fill_table_front_structure(
                                _id=_id,
                                graph_name=f"Выход «{out}» - Отчет по классам - {type_name} выборка",
                                plot_data=report
                            )
                        )
                        _id += 1
            return return_data
        except Exception as e:
            print_error(BaseClassificationCallback().name, method_name, e)

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config):
        method_name = 'balance_data_request'
        try:
            return_data = []
            _id = 0
            for out in options.data.outputs.keys():
                for class_type in dataset_balance.get(f"{out}").keys():
                    preset = {}
                    for data_type in ['train', 'val']:
                        class_names, class_count = sort_dict(
                            dict_to_sort=dataset_balance.get(f"{out}").get(class_type).get(data_type),
                            mode=interactive_config.data_balance.sorted.name
                        )
                        preset[data_type] = fill_graph_front_structure(
                            _id=_id,
                            _type='histogram',
                            type_data=data_type,
                            graph_name=f"Выход {out} - "
                                       f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка",
                            short_name=f"{out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'}",
                            x_label="Название класса",
                            y_label="Значение",
                            plot_data=[fill_graph_plot_data(x=class_names, y=class_count)],
                        )
                        _id += 1
                    return_data.append(preset)
            return return_data
        except Exception as e:
            print_error(BaseClassificationCallback().name, method_name, e)

    @staticmethod
    def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                    choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.best,
                                    seed_idx: list = None) -> dict:
        method_name = 'prepare_example_idx_to_show'
        try:
            example_idx = []
            encoding = options.data.outputs.get(output).encoding
            if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
                if array.shape[-1] == true_array.shape[-1] and encoding == \
                        LayerEncodingChoice.ohe and true_array.shape[-1] > 1:
                    classes = np.argmax(true_array, axis=-1)
                elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[-1] > 1:
                    classes = copy.deepcopy(true_array)
                elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[-1] == 1:
                    classes = copy.deepcopy(true_array)
                else:
                    classes = copy.deepcopy(true_array)
                class_idx = {}
                for _id in range(options.data.outputs.get(output).num_classes):
                    class_idx[_id] = {}
                for i, pred in enumerate(array):
                    class_idx[classes[i]][i] = pred[classes[i]]
                for _id in range(options.data.outputs.get(output).num_classes):
                    class_idx[_id] = list(sort_dict(class_idx[_id], mode=BalanceSortedChoice.ascending)[0])
                num_ex = copy.deepcopy(count)
                while num_ex:
                    stop = False
                    while not stop:
                        key = np.random.choice(list(class_idx.keys()))
                        if class_idx[key]:
                            stop = True
                    if choice_type == ExampleChoiceTypeChoice.best:
                        example_idx.append(class_idx[key][-1])
                        class_idx[key].pop(-1)
                    if class_idx[key] and choice_type == ExampleChoiceTypeChoice.worst:
                        example_idx.append(class_idx[key][0])
                        class_idx[key].pop(0)
                    num_ex -= 1

            elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
                example_idx = seed_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.random:
                true_id = []
                false_id = []
                for i, ex in enumerate(true_array):
                    if np.argmax(ex, axis=-1) == np.argmax(array[i], axis=-1):
                        true_id.append(i)
                    else:
                        false_id.append(i)
                np.random.shuffle(true_id)
                np.random.shuffle(false_id)
                true_false_dict = {'true': true_id, 'false': false_id}

                for _ in range(count):
                    if true_false_dict.get('true') and true_false_dict.get('false'):
                        key = np.random.choice(list(true_false_dict.keys()))
                    elif true_false_dict.get('true') and not true_false_dict.get('false'):
                        key = 'true'
                    else:
                        key = 'false'
                    example_idx.append(true_false_dict.get(key)[0])
                    true_false_dict.get(key).pop(0)
                np.random.shuffle(example_idx)
            else:
                pass
            return example_idx
        except Exception as e:
            print_error(BaseClassificationCallback().name, method_name, e)

    @staticmethod
    def postprocess_classification(predict_array: np.ndarray, true_array: np.ndarray, options: DatasetOutputsData,
                                   show_stat: bool = False, return_mode='deploy'):
        method_name = 'postprocess_classification'
        try:
            labels = options.classes_names
            ohe = True if options.encoding == LayerEncodingChoice.ohe else False
            actual_value = np.argmax(true_array, axis=-1) if ohe else true_array
            data = {
                "y_true": {},
                "y_pred": {},
                "stat": {}
            }
            if return_mode == 'deploy':
                labels_from_array = []
                for class_idx in predict_array:
                    class_dist = sorted(class_idx, reverse=True)
                    labels_dist = []
                    for j in class_dist:
                        labels_dist.append((labels[list(class_idx).index(j)], round(float(j) * 100, 1)))
                    labels_from_array.append(labels_dist)
                return labels[actual_value], labels_from_array

            if return_mode == 'callback':
                data["y_true"] = {
                    "type": "str",
                    "data": [
                        {
                            "title": "Класс",
                            "value": labels[actual_value],
                            "color_mark": None
                        }
                    ]
                }
                if labels[actual_value] == labels[np.argmax(predict_array)]:
                    color_mark = 'success'
                else:
                    color_mark = 'wrong'
                data["y_pred"] = {
                    "type": "str",
                    "data": [
                        {
                            "title": "Класс",
                            "value": labels[np.argmax(predict_array)],
                            "color_mark": color_mark
                        }
                    ]
                }
                if show_stat:
                    data["stat"] = {
                        "type": "str",
                        "data": []
                    }
                    for i, val in enumerate(predict_array):
                        if val == max(predict_array) and labels[i] == labels[actual_value]:
                            class_color_mark = "success"
                        elif val == max(predict_array) and labels[i] != labels[actual_value]:
                            class_color_mark = "wrong"
                        else:
                            class_color_mark = None
                        data["stat"]["data"].append(
                            {
                                'title': labels[i],
                                'value': f"{round(val * 100, 1)}%",
                                'color_mark': class_color_mark
                            }
                        )
                return data
        except Exception as e:
            print_error(BaseClassificationCallback().name, method_name, e)

    @staticmethod
    def prepare_class_idx(y_true: dict, options) -> dict:
        method_name = 'prepare_class_idx'
        try:
            class_idx = {}
            for data_type in y_true.keys():
                class_idx[data_type] = {}
                for out in y_true.get(data_type).keys():
                    class_idx[data_type][out] = {}
                    ohe = options.data.outputs.get(int(out)).encoding == LayerEncodingChoice.ohe
                    for name in options.data.outputs.get(int(out)).classes_names:
                        class_idx[data_type][out][name] = []
                    y_true_argmax = np.argmax(y_true.get(data_type).get(out), axis=-1) if ohe \
                        else np.squeeze(y_true.get(data_type).get(out))
                    for idx in range(len(y_true_argmax)):
                        class_idx[data_type][out][
                            options.data.outputs.get(int(out)).classes_names[y_true_argmax[idx]]].append(idx)
            return class_idx
        except Exception as e:
            print_error(BaseClassificationCallback().name, method_name, e)


class ImageClassificationCallback(BaseClassificationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'ImageClassificationCallback'
        # print(f'Callback {self.name} is called')

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            x_val = {}
            inverse_x_val = {}
            if options.data.group == DatasetGroupChoice.keras:
                for data_type in ['train', 'val']:
                    x_val[data_type] = options.X.get(data_type)
            return x_val, inverse_x_val
        except Exception as e:
            print_error(ImageClassificationCallback().name, method_name, e)

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
        except Exception as e:
            print_error(ImageClassificationCallback().name, method_name, e)

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        method_name = 'postprocess_deploy'
        try:
            x_array, inverse_x_array = ImageClassificationCallback.get_x_array(options)
            return_data = {}
            if array is None:
                print("postprocess_deploy: array is None")

            for i, output_id in enumerate(options.data.outputs.keys()):
                true_array = get_y_true(options, output_id)
                if len(options.data.outputs.keys()) > 1:
                    postprocess_array = array[i]
                else:
                    postprocess_array = array
                example_idx = ImageClassificationCallback.prepare_example_idx_to_show(
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
                    # preset_path = os.path.join(save_path, "deploy_presets")
                    source = ImageClassificationCallback.postprocess_initial_source(
                        options=options,
                        input_id=input_id,
                        save_id=_id,
                        example_id=idx,
                        dataset_path=dataset_path,
                        preset_path=save_path,
                        x_array=None if not x_array else x_array.get(f"{input_id}"),
                        return_mode='deploy'
                    )
                    actual_value, predict_values = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(ImageClassificationCallback().name, method_name, e)

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors, raw_y_pred=None):
        method_name = 'intermediate_result_request'
        try:
            print(method_name)
            return_data = {}
            if interactive_config.intermediate_result.show_results:
                data_type = interactive_config.intermediate_result.data_type.name
                # print('\ndata_type', data_type, '\n')
                for idx in range(interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': None,
                        'statistic_values': {}
                    }
                    for inp in options.data.inputs.keys():
                        data = ImageClassificationCallback.postprocess_initial_source(
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
                        data = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(ImageClassificationCallback().name, method_name, e)


class TextClassificationCallback(BaseClassificationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'TextClassificationCallback'
        # print(f'Callback {self.name} is called')

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            x_val = None
            inverse_x_val = None
            return x_val, inverse_x_val
        except Exception as e:
            print_error(TextClassificationCallback().name, method_name, e)

    @staticmethod
    def postprocess_initial_source(options, example_id: int, return_mode='deploy', data_type='val'):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            for inp in options.data.inputs.keys():
                for column_name in options.dataframe.get(data_type).columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get(data_type).columns.tolist().index(column_name))
            data = []
            source = ""
            for column in column_idx:
                source = options.dataframe.get(data_type).iat[example_id, column]
                if return_mode == 'deploy':
                    break
                if return_mode == 'callback':
                    title = "Текст"
                    data = [
                        {
                            "title": title,
                            "value": source,
                            "color_mark": None
                        }
                    ]
            if return_mode == 'deploy':
                return source
            if return_mode == 'callback':
                return data
        except Exception as e:
            print_error(TextClassificationCallback().name, method_name, e)

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
                example_idx = ImageClassificationCallback.prepare_example_idx_to_show(
                    array=postprocess_array[:len(array)],
                    true_array=true_array[:len(array)],
                    options=options,
                    output=output_id,
                    count=int(len(array) * DEPLOY_PRESET_PERCENT / 100)
                )

                return_data[output_id] = []
                for idx in example_idx:
                    source = TextClassificationCallback.postprocess_initial_source(
                        options=options,
                        example_id=idx,
                        return_mode='deploy'
                    )
                    actual_value, predict_values = ImageClassificationCallback.postprocess_classification(
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
            return return_data
        except Exception as e:
            print_error(TextClassificationCallback().name, method_name, e)

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
                        data = TextClassificationCallback.postprocess_initial_source(
                            options=options,
                            example_id=example_idx[idx],
                            return_mode='callback',
                            data_type=data_type
                        )
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'type': 'text',
                            'data': data,
                        }
                    for out in options.data.outputs.keys():
                        data = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(TextClassificationCallback().name, method_name, e)


class DataframeClassificationCallback(BaseClassificationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'DataframeClassificationCallback'
        # print(f'Callback {self.name} is called')

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            inverse_x_val = None
            if not options.data.use_generator:
                x_val = options.X.get("val")
            else:
                x_val = {}
                for inp in options.dataset['val'].keys():
                    x_val[inp] = []
                    for x_val_, _ in options.dataset['val'].batch(1):
                        x_val[inp].extend(x_val_.get(f'{inp}').numpy())
                    x_val[inp] = np.array(x_val[inp])
            return x_val, inverse_x_val
        except Exception as e:
            print_error(DataframeClassificationCallback().name, method_name, e)

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, return_mode='deploy', data_type='val'):
        method_name = 'postprocess_initial_source'
        try:
            data = []
            source = []
            for col_name in options.data.columns.get(input_id).keys():
                value = options.dataframe.get(data_type)[col_name].to_list()[example_id]
                if return_mode == 'deploy':
                    source.append(value)
                if return_mode == 'callback':
                    data.append(
                        {
                            "title": col_name.split("_", 1)[-1],
                            "value": value,
                            "color_mark": None
                        }
                    )

            if return_mode == 'deploy':
                return source
            if return_mode == 'callback':
                return data
        except Exception as e:
            print_error(DataframeClassificationCallback().name, method_name, e)

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
                example_idx = ImageClassificationCallback.prepare_example_idx_to_show(
                    array=postprocess_array[:len(array)],
                    true_array=true_array[:len(array)],
                    options=options,
                    output=output_id,
                    count=int(len(array) * DEPLOY_PRESET_PERCENT / 100)
                )
                return_data[output_id] = []
                for idx in example_idx:
                    input_id = list(options.data.inputs.keys())[0]
                    source = DataframeClassificationCallback.postprocess_initial_source(
                        options=options,
                        input_id=input_id,
                        example_id=idx,
                        return_mode='deploy'
                    )
                    actual_value, predict_values = ImageClassificationCallback.postprocess_classification(
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
            return return_data
        except Exception as e:
            print_error(DataframeClassificationCallback().name, method_name, e)

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
                        data = DataframeClassificationCallback.postprocess_initial_source(
                            options=options,
                            input_id=inp,
                            example_id=example_idx[idx],
                            return_mode='callback',
                            data_type=data_type
                        )
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'type': 'str',
                            'data': data,
                        }

                    for out in options.data.outputs.keys():
                        data = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(DataframeClassificationCallback().name, method_name, e)


class AudioClassificationCallback(BaseClassificationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'AudioClassificationCallback'
        # print(f'Callback {self.name} is called')

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            x_val = None
            inverse_x_val = None
            return x_val, inverse_x_val
        except Exception as e:
            print_error(AudioClassificationCallback().name, method_name, e)

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, dataset_path: str,
                                   preset_path: str, save_id: int = None, return_mode='deploy', data_type='val'):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            for inp in options.data.inputs.keys():
                for column_name in options.dataframe.get(data_type).columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get(data_type).columns.tolist().index(column_name))
            initial_file_path = os.path.join(
                dataset_path, options.dataframe.get(data_type).iat[example_id, column_idx[0]]
            )
            if not save_id:
                return str(os.path.abspath(initial_file_path))

            if return_mode == 'callback':
                source = os.path.join(preset_path, f"initial_data_audio_{save_id}_input_{input_id}.webm")
                AudioSegment.from_file(initial_file_path).export(source, format="webm")
                data = [
                    {
                        "title": "Аудио",
                        "value": source,
                        "color_mark": None
                    }
                ]
                return data
            if return_mode == 'deploy':
                source = os.path.join(preset_path, "deploy_presets",
                                      f"initial_data_audio_{save_id}_input_{input_id}.webm")
                return_source = os.path.join("deploy_presets", f"initial_data_audio_{save_id}_input_{input_id}.webm")
                AudioSegment.from_file(initial_file_path).export(source, format="webm")
                return return_source
        except Exception as e:
            print_error(AudioClassificationCallback().name, method_name, e)

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
                example_idx = ImageClassificationCallback.prepare_example_idx_to_show(
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
                    source = AudioClassificationCallback.postprocess_initial_source(
                        options=options,
                        input_id=input_id,
                        save_id=_id,
                        example_id=idx,
                        dataset_path=dataset_path,
                        preset_path=save_path,
                        return_mode='deploy'
                    )
                    actual_value, predict_values = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(AudioClassificationCallback().name, method_name, e)

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
                        data = AudioClassificationCallback.postprocess_initial_source(
                            options=options,
                            input_id=inp,
                            save_id=idx + 1,
                            example_id=example_idx[idx],
                            dataset_path=dataset_path,
                            preset_path=preset_path,
                            return_mode='callback',
                            data_type=data_type
                        )
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'type': 'audio',
                            'data': data,
                        }

                    for out in options.data.outputs.keys():
                        data = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(AudioClassificationCallback().name, method_name, e)


class VideoClassificationCallback(BaseClassificationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'VideoClassificationCallback'
        # print(f'Callback {self.name} is called')

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            x_val = None
            inverse_x_val = None
            return x_val, inverse_x_val
        except Exception as e:
            print_error(VideoClassificationCallback().name, method_name, e)

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, dataset_path: str, preset_path: str,
                                   save_id: int = None, return_mode='deploy', data_type='val'):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            for inp in options.data.inputs.keys():
                for column_name in options.dataframe.get(data_type).columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get(data_type).columns.tolist().index(column_name))
            initial_file_path = os.path.join(
                dataset_path, options.dataframe.get(data_type).iat[example_id, column_idx[0]]
            )
            if not save_id:
                return str(os.path.abspath(initial_file_path))

            clip = moviepy_editor.VideoFileClip(initial_file_path)

            if return_mode == 'deploy':
                source = os.path.join(preset_path, "deploy_presets",
                                      f"initial_data_video_{save_id}_input_{input_id}.webm")
                return_source = os.path.join("deploy_presets", f"initial_data_video_{save_id}_input_{input_id}.webm")
                clip.write_videofile(source)
                return return_source
            if return_mode == 'callback':
                source = os.path.join(preset_path, f"initial_data_video_{save_id}_input_{input_id}.webm")
                clip.write_videofile(source)
                data = [
                    {
                        "title": "Видео",
                        "value": source,
                        "color_mark": None
                    }
                ]
                return data
        except Exception as e:
            print_error(VideoClassificationCallback().name, method_name, e)

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
                example_idx = ImageClassificationCallback.prepare_example_idx_to_show(
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
                    source = VideoClassificationCallback.postprocess_initial_source(
                        options=options,
                        input_id=input_id,
                        save_id=_id,
                        example_id=idx,
                        dataset_path=dataset_path,
                        preset_path=save_path,
                        return_mode='deploy'
                    )
                    actual_value, predict_values = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(VideoClassificationCallback().name, method_name, e)

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
                        data = VideoClassificationCallback.postprocess_initial_source(
                            options=options,
                            input_id=inp,
                            save_id=idx + 1,
                            example_id=example_idx[idx],
                            dataset_path=dataset_path,
                            preset_path=preset_path,
                            return_mode='callback',
                            data_type=data_type
                        )
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'type': 'video',
                            'data': data,
                        }
                    for out in options.data.outputs.keys():
                        data = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(VideoClassificationCallback().name, method_name, e)


class TimeseriesTrendCallback(BaseClassificationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'TimeseriesTrendCallback'
        # print(f'Callback {self.name} is called')

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            inverse_x_val = {}
            if not options.data.use_generator:
                x_val = options.X.get("val")
            else:
                x_val = {}
                for inp in options.dataset['val'].keys():
                    x_val[inp] = []
                    for x_val_, _ in options.dataset['val'].batch(1):
                        x_val[inp].extend(x_val_.get(f'{inp}').numpy())
                    x_val[inp] = np.array(x_val[inp])
            for inp in x_val.keys():
                preprocess_dict = options.preprocessing.preprocessing.get(int(inp))
                inverse_x = np.zeros_like(x_val.get(inp)[:, :, 0:1])
                for i, column in enumerate(preprocess_dict.keys()):
                    if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                        _options = {int(inp): {column: x_val.get(inp)[:, :, i]}}
                        inverse_col = np.expand_dims(
                            options.preprocessing.inverse_data(_options).get(int(inp)).get(column), axis=-1
                        )
                    else:
                        inverse_col = x_val.get(inp)[:, :, i:i + 1]
                    inverse_x = np.concatenate([inverse_x, inverse_col], axis=-1)
                inverse_x_val[inp] = inverse_x[:, :, 1:]
            return x_val, inverse_x_val
        except Exception as e:
            print_error(TimeseriesTrendCallback().name, method_name, e)

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, inverse_x_array=None, data_type='val'):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            for inp in options.data.inputs.keys():
                for column_name in options.dataframe.get(data_type).columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get(data_type).columns.tolist().index(column_name))
            graphics_data = []
            names = ""
            multi = False
            inverse_x = inverse_x_array.get(f'{input_id}')
            for i, channel in enumerate(options.data.columns.get(input_id).keys()):
                multi = True if i > 0 else False
                names += f"«{channel.split('_', 1)[-1]}», "
                length = len(inverse_x) if len(inverse_x) < MAX_GRAPH_LENGTH else MAX_GRAPH_LENGTH
                graphics_data.append(
                    fill_graph_front_structure(
                        _id=i + 1,
                        _type='graphic',
                        graph_name=f"График канала «{channel.split('_', 1)[-1]}»",
                        short_name=f"«{channel.split('_', 1)[-1]}»",
                        x_label="Время",
                        y_label="Значение",
                        plot_data=[
                            fill_graph_plot_data(
                                label="Исходное значение",
                                x=np.arange(inverse_x[example_id].shape[-2]).astype('int').tolist()[-length:],
                                y=inverse_x[example_id][:, i].astype('float').tolist()[-length:]
                            )
                        ],
                    )
                )
            data = [
                {
                    "title": f"График{'и' if multi else ''} по канал{'ам' if multi else 'у'} {names[:-2]}",
                    "value": graphics_data,
                    "color_mark": None
                }
            ]
            return data
        except Exception as e:
            print_error(TimeseriesTrendCallback().name, method_name, e)

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        method_name = 'postprocess_deploy'
        try:
            x_array, inverse_x_array = TimeseriesTrendCallback().get_x_array(options)
            return_data = {}
            for i, output_id in enumerate(options.data.outputs.keys()):
                true_array = get_y_true(options, output_id)
                if len(options.data.outputs.keys()) > 1:
                    postprocess_array = array[i]
                else:
                    postprocess_array = array
                example_idx = ImageClassificationCallback.prepare_example_idx_to_show(
                    array=postprocess_array[:len(array)],
                    true_array=true_array[:len(array)],
                    options=options,
                    output=output_id,
                    count=int(len(array) * DEPLOY_PRESET_PERCENT / 100)
                )
                return_data[output_id] = []
                for idx in example_idx:
                    data = {
                        'source': {},
                        'predict': {}
                    }
                    for inp in options.data.inputs.keys():
                        for k, inp_col in enumerate(options.data.columns.get(inp).keys()):
                            data['source'][inp_col.split('_', 1)[-1]] = \
                                round_list(list(inverse_x_array[f"{inp}"][idx][:, k]))
                    for channel in options.data.columns.get(output_id).keys():
                        _, predict_values = ImageClassificationCallback.postprocess_classification(
                            predict_array=np.expand_dims(postprocess_array[idx], axis=0),
                            true_array=true_array[idx],
                            options=options.data.outputs[output_id],
                            return_mode='deploy'
                        )
                        data['predict'][channel.split('_', 1)[-1]] = \
                            [data['source'][channel.split('_', 1)[-1]], [predict_values[0][0][0]]]
                        # data['predict'][channel.split('_', 1)[-1]] = \
                        #     [data['source'][channel.split('_', 1)[-1]], predict_values[0]]
                    return_data[output_id].append(data)
            return return_data
        except Exception as e:
            print_error(TimeseriesTrendCallback().name, method_name, e)

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
                        data = TimeseriesTrendCallback.postprocess_initial_source(
                            options=options,
                            input_id=inp,
                            example_id=example_idx[idx],
                            inverse_x_array=inverse_x_val,
                            data_type=data_type
                        )
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'type': 'graphic',
                            'data': data,
                        }
                    for out in options.data.outputs.keys():
                        data = ImageClassificationCallback.postprocess_classification(
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
        except Exception as e:
            print_error(TimeseriesTrendCallback().name, method_name, e)
