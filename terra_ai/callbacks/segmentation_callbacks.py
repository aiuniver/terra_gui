import colorsys
import copy
import os
from typing import Optional

import matplotlib
import numpy as np
from PIL import Image
from pandas import DataFrame
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.callbacks.utils import dice_coef, sort_dict, get_y_true, get_image_class_colormap, get_confusion_matrix, \
    fill_heatmap_front_structure, get_classification_report, fill_table_front_structure, fill_graph_front_structure, \
    fill_graph_plot_data, print_error, segmentation_metric
from terra_ai.data.datasets.dataset import DatasetOutputsData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerEncodingChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.settings import CALLBACK_CLASSIFICATION_TREASHOLD_VALUE, DEPLOY_PRESET_PERCENT


class BaseSegmentationCallback:
    def __init__(self):
        self.name = 'BaseSegmentationCallback'

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            x_val = None
            inverse_x_val = None
            return x_val, inverse_x_val
        except Exception as e:
            print_error(BaseSegmentationCallback().name, method_name, e)

    @staticmethod
    def get_y_true(options, dataset_path):
        method_name = 'get_y_true'
        try:
            y_true = {"train": {}, "val": {}}
            inverse_y_true = {"train": {}, "val": {}}
            for data_type in y_true.keys():
                for out in options.data.outputs.keys():
                    if not options.data.use_generator:
                        y_true[data_type][f"{out}"] = options.Y.get(data_type).get(f"{out}")
                    else:
                        y_true[data_type][f"{out}"] = []
                        for _, y_val in options.dataset[data_type].batch(1):
                            y_true[data_type][f"{out}"].extend(y_val.get(f'{out}').numpy())
                        y_true[data_type][f"{out}"] = np.array(y_true[data_type][f"{out}"])
            return y_true, inverse_y_true
        except Exception as e:
            print_error(BaseSegmentationCallback().name, method_name, e)

    @staticmethod
    def get_y_pred(y_true, y_pred, options):
        method_name = 'get_y_pred'
        try:
            reformat_pred = {}
            inverse_pred = {}
            for idx, out in enumerate(y_true.get('val').keys()):
                if len(y_true.get('val').keys()) == 1:
                    reformat_pred[out] = y_pred
                else:
                    reformat_pred[out] = y_pred[idx]
            return reformat_pred, inverse_pred
        except Exception as e:
            print_error(BaseSegmentationCallback().name, method_name, e)

    @staticmethod
    def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                    choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.best,
                                    seed_idx: list = None) -> dict:
        method_name = 'prepare_example_idx_to_show'
        try:
            example_idx = []
            encoding = options.data.outputs.get(output).encoding
            if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
                if encoding == LayerEncodingChoice.ohe:
                    array = to_categorical(
                        np.argmax(array, axis=-1), num_classes=options.data.outputs.get(output).num_classes
                    )
                if encoding == LayerEncodingChoice.multi:
                    array = np.where(array >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100, 1., 0.)
                dice_val = dice_coef(true_array, array, batch_mode=True)
                dice_dict = dict(zip(np.arange(0, len(dice_val)), dice_val))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = sort_dict(dice_dict, mode=BalanceSortedChoice.descending)
                    example_idx = example_idx[:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx, _ = sort_dict(dice_dict, mode=BalanceSortedChoice.ascending)
                    example_idx = example_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
                example_idx = seed_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.random:
                if encoding == LayerEncodingChoice.ohe:
                    array = to_categorical(
                        np.argmax(array, axis=-1),
                        num_classes=options.data.outputs.get(output).num_classes
                    )
                if encoding == LayerEncodingChoice.multi:
                    array = np.where(array >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100, 1, 0)
                dice_val = dice_coef(true_array, array, batch_mode=True)
                true_id = []
                false_id = []
                for i, ex in enumerate(dice_val):
                    if ex >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100:
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
                example_idx = np.random.randint(0, len(true_array), count)

            return example_idx
        except Exception as e:
            print_error(BaseSegmentationCallback().name, method_name, e)


class ImageSegmentationCallback(BaseSegmentationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'ImageSegmentationCallback'
        print(f'Callback {self.name} is called')

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, dataset_path: str, preset_path: str,
                                   save_id: int = None, return_mode='deploy'):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            for inp in options.data.inputs.keys():
                for column_name in options.dataframe.get('val').columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))
            initial_file_path = os.path.join(
                dataset_path, options.dataframe.get('val').iat[example_id, column_idx[0]]
            )
            if not save_id:
                return str(os.path.abspath(initial_file_path))

            img = Image.open(initial_file_path)
            img = img.resize(options.data.inputs.get(input_id).shape[0:2][::-1], Image.ANTIALIAS)
            img = img.convert('RGB')
            source = os.path.join(preset_path, f"initial_data_image_{save_id}_input_{input_id}.webp")
            img.save(source, 'webp')
            if return_mode == 'deploy':
                return source
            if return_mode == 'callback':
                data = [
                    {
                        "title": "Изображение",
                        "value": source,
                        "color_mark": None
                    }
                ]
                return data
        except Exception as e:
            print_error(ImageSegmentationCallback().name, method_name, e)

    @staticmethod
    def postprocess_segmentation(predict_array: np.ndarray, true_array: Optional[np.ndarray],
                                 options: DatasetOutputsData, output_id: int, image_id: int, save_path: str,
                                 colors: list = None, return_mode='deploy', show_stat: bool = False):
        method_name = 'postprocess_segmentation'
        try:
            data = {
                "y_true": {},
                "y_pred": {},
                "stat": {}
            }
            if return_mode == 'deploy':
                array = np.expand_dims(np.argmax(predict_array, axis=-1), axis=-1) * 512
                for i, color in enumerate(options.classes_colors):
                    array = np.where(
                        array == i * 512,
                        np.array(color.as_rgb_tuple()),
                        array
                    )
                array = array.astype("uint8")
                img_save_path = os.path.join(
                    save_path,
                    f"image_segmentation_postprocessing_{image_id}_output_{output_id}.webp"
                )
                matplotlib.image.imsave(img_save_path, array)
                return img_save_path

            if return_mode == 'callback':
                y_true = np.expand_dims(np.argmax(true_array, axis=-1), axis=-1) * 512
                for i, color in enumerate(colors):
                    y_true = np.where(y_true == i * 512, np.array(color), y_true)
                y_true = y_true.astype("uint8")
                y_true_save_path = os.path.join(
                    save_path,
                    f"true_segmentation_data_image_{image_id}_output_{output_id}.webp"
                )
                matplotlib.image.imsave(y_true_save_path, y_true)

                data["y_true"] = {
                    "type": "image",
                    "data": [
                        {
                            "title": "Изображение",
                            "value": y_true_save_path,
                            "color_mark": None
                        }
                    ]
                }

                y_pred = np.expand_dims(np.argmax(predict_array, axis=-1), axis=-1) * 512
                for i, color in enumerate(colors):
                    y_pred = np.where(y_pred == i * 512, np.array(color), y_pred)
                y_pred = y_pred.astype("uint8")
                y_pred_save_path = os.path.join(
                    save_path,
                    f"predict_segmentation_data_image_{image_id}_output_{output_id}.webp"
                )
                matplotlib.image.imsave(y_pred_save_path, y_pred)
                data["y_pred"] = {
                    "type": "image",
                    "data": [
                        {
                            "title": "Изображение",
                            "value": y_pred_save_path,
                            "color_mark": None
                        }
                    ]
                }
                if show_stat:
                    data["stat"] = {
                        "type": "str",
                        "data": []
                    }
                    y_true = np.array(true_array).astype('int')
                    y_pred = to_categorical(np.argmax(predict_array, axis=-1), options.num_classes).astype('int')
                    count = 0
                    mean_val = 0
                    for idx, cls in enumerate(options.classes_names):
                        dice_val = np.round(
                            dice_coef(y_true[:, :, idx], y_pred[:, :, idx], batch_mode=False) * 100, 1)
                        count += 1
                        mean_val += dice_val
                        data["stat"]["data"].append(
                            {
                                'title': cls,
                                'value': f"{dice_val}%",
                                'color_mark': 'success' if dice_val >= 90 else 'wrong'
                            }
                        )
                    data["stat"]["data"].insert(
                        0,
                        {
                            'title': "Средняя точность",
                            'value': f"{round(mean_val / count, 2)}%",
                            'color_mark': 'success' if mean_val / count >= 90 else 'wrong'
                        }
                    )
                return data
        except Exception as e:
            print_error(ImageSegmentationCallback().name, method_name, e)

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
                example_idx = ImageSegmentationCallback().prepare_example_idx_to_show(
                    array=postprocess_array[:len(array)],
                    true_array=true_array[:len(array)],
                    options=options,
                    output=output_id,
                    count=int(len(array) * DEPLOY_PRESET_PERCENT / 100)
                )
                return_data[output_id] = []
                data = []
                for j, cls in enumerate(options.data.outputs.get(output_id).classes_names):
                    data.append((cls, options.data.outputs.get(output_id).classes_colors[j].as_rgb_tuple()))
                _id = 1
                for idx in example_idx:
                    input_id = list(options.data.inputs.keys())[0]
                    return_data[output_id].append(
                        {
                            "source": ImageSegmentationCallback.postprocess_initial_source(
                                options=options,
                                input_id=input_id,
                                save_id=_id,
                                example_id=idx,
                                dataset_path=dataset_path,
                                preset_path=save_path,
                                return_mode='deploy'
                            ),
                            "segment": ImageSegmentationCallback().postprocess_segmentation(
                                predict_array=array[idx],
                                true_array=None,
                                options=options.data.outputs.get(output_id),
                                output_id=output_id,
                                image_id=idx,
                                save_path=save_path,
                                return_mode='deploy'
                            ),
                            "data": data
                        }
                    )
                    _id += 1
            return return_data
        except Exception as e:
            print_error(ImageSegmentationCallback().name, method_name, e)

    @staticmethod
    def dataset_balance(options, y_true, preset_path: str, class_colors) -> dict:
        method_name = 'dataset_balance'
        try:
            dataset_balance = {}
            for out in options.data.outputs.keys():
                dataset_balance[f"{out}"] = {
                    "presence_balance": {}, "square_balance": {}, "colormap": {}
                }
                for data_type in ['train', 'val']:
                    dataset_balance[f"{out}"]["colormap"][data_type] = {}
                    classes_names = options.data.outputs.get(out).classes_names
                    classes = np.arange(options.data.outputs.get(out).num_classes)
                    class_percent = {}
                    class_count = {}
                    for cl in classes:
                        class_percent[classes_names[cl]] = np.round(
                            np.sum(
                                y_true.get(data_type).get(f"{out}")[..., cl]) * 100
                            / np.prod(y_true.get(data_type).get(f"{out}")[..., 0].shape)
                        ).astype("float").tolist()
                        class_count[classes_names[cl]] = 0
                        colormap_path = os.path.join(
                            preset_path, f"balance_segmentation_colormap_{data_type}_class_{classes_names[cl]}.webp"
                        )
                        get_image_class_colormap(
                            array=y_true.get(data_type).get(f"{out}"),
                            colors=class_colors,
                            class_id=cl,
                            save_path=colormap_path
                        )
                        dataset_balance[f"{out}"]["colormap"][data_type][classes_names[cl]] = colormap_path

                    for img_array in np.argmax(y_true.get(data_type).get(f"{out}"), axis=-1):
                        for cl in classes:
                            if cl in img_array:
                                class_count[classes_names[cl]] += 1
                    dataset_balance[f"{out}"]["presence_balance"][data_type] = class_count
                    dataset_balance[f"{out}"]["square_balance"][data_type] = class_percent
            return dataset_balance
        except Exception as e:
            print_error(ImageSegmentationCallback().name, method_name, e)

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors, raw_y_pred) -> dict:
        method_name = 'intermediate_result_request'
        try:
            return_data = {}
            if interactive_config.intermediate_result.show_results:
                for idx in range(interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': {},
                        'statistic_values': {}
                    }
                    if not len(options.data.outputs.keys()) == 1:
                        for inp in options.data.inputs.keys():
                            data = ImageSegmentationCallback.postprocess_initial_source(
                                options=options,
                                input_id=inp,
                                save_id=idx + 1,
                                example_id=example_idx[idx],
                                dataset_path=dataset_path,
                                preset_path=preset_path,
                                return_mode='callback'
                            )
                            return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                                'type': 'image', 'data': data,
                            }
                    for out in options.data.outputs.keys():
                        data = ImageSegmentationCallback().postprocess_segmentation(
                            predict_array=y_pred.get(f'{out}')[example_idx[idx]],
                            true_array=y_true.get('val').get(f'{out}')[example_idx[idx]],
                            options=options.data.outputs.get(out),
                            colors=class_colors,
                            output_id=out,
                            image_id=idx,
                            save_path=preset_path,
                            return_mode='callback',
                            show_stat=interactive_config.intermediate_result.show_statistic
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
            print_error(ImageSegmentationCallback().name, method_name, e)

    @staticmethod
    def statistic_data_request(interactive_config, options, y_true, inverse_y_true,
                               y_pred, inverse_y_pred, raw_y_pred=None) -> list:
        method_name = 'statistic_data_request'
        try:
            return_data = []
            _id = 1
            for out in interactive_config.statistic_data.output_id:
                cm, cm_percent = get_confusion_matrix(
                    np.argmax(y_true.get("val").get(f"{out}"), axis=-1).reshape(
                        np.prod(np.argmax(y_true.get("val").get(f"{out}"), axis=-1).shape)).astype('int'),
                    np.argmax(y_pred.get(f'{out}'), axis=-1).reshape(
                        np.prod(np.argmax(y_pred.get(f'{out}'), axis=-1).shape)).astype('int'),
                    get_percent=True
                )
                return_data.append(
                    fill_heatmap_front_structure(
                        _id=_id,
                        _type="heatmap",
                        graph_name=f"Выходной слой «{out}» - Confusion matrix",
                        short_name=f"{out} - Confusion matrix",
                        x_label="Предсказание",
                        y_label="Истинное значение",
                        labels=options.data.outputs.get(out).classes_names,
                        data_array=cm,
                        data_percent_array=cm_percent,
                    )
                )
                _id += 1
            return return_data
        except Exception as e:
            print_error(ImageSegmentationCallback().name, method_name, e)

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config) -> list:
        method_name = 'balance_data_request'
        try:
            return_data = []
            _id = 0
            for out in options.data.outputs.keys():
                for class_type in dataset_balance.get(f"{out}").keys():
                    preset = {}
                    if class_type in ["presence_balance", "square_balance"]:
                        for data_type in ['train', 'val']:
                            names, count = sort_dict(
                                dict_to_sort=dataset_balance.get(f"{out}").get(class_type).get(data_type),
                                mode=interactive_config.data_balance.sorted.name
                            )
                            preset[data_type] = fill_graph_front_structure(
                                _id=_id,
                                _type='histogram',
                                type_data=data_type,
                                graph_name=f"Выход {out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - "
                                           f"{'баланс присутсвия' if class_type == 'presence_balance' else 'процент пространства'}",
                                short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                           f"{'присутсвие' if class_type == 'presence_balance' else 'пространство'}",
                                x_label="Название класса",
                                y_label="Значение",
                                plot_data=[fill_graph_plot_data(x=names, y=count)],
                            )
                            _id += 1
                        return_data.append(preset)
                    if class_type == "colormap":
                        for class_name, map_link in dataset_balance.get(f"{out}").get('colormap').get(
                                'train').items():
                            preset = {}
                            for data_type in ['train', 'val']:
                                preset[data_type] = fill_graph_front_structure(
                                    _id=_id,
                                    _type='colormap',
                                    type_data=data_type,
                                    graph_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка "
                                               f"- Цветовая карта класса {class_name}",
                                    short_name="",
                                    x_label="",
                                    y_label="",
                                    plot_data=map_link,
                                )
                                _id += 1
                            return_data.append(preset)
            return return_data
        except Exception as e:
            print_error(ImageSegmentationCallback().name, method_name, e)


class TextSegmentationCallback(BaseSegmentationCallback):
    def __init__(self):
        super().__init__()
        self.name = 'ImageSegmentationCallback'
        print(f'Callback {self.name} is called')

    @staticmethod
    def postprocess_initial_source(options, example_id: int, return_mode='deploy'):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            for inp in options.data.inputs.keys():
                for column_name in options.dataframe.get('val').columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))
            data = []
            source = ""
            for column in column_idx:
                source = options.dataframe.get('val').iat[example_id, column]
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
            print_error(TextSegmentationCallback().name, method_name, e)

    @staticmethod
    def postprocess_text_segmentation(pred_array: np.ndarray, options: DatasetOutputsData, dataframe: DataFrame,
                                      dataset_params: dict, example_id: int, return_mode='deploy',
                                      class_colors: list = None, show_stat: bool = False,
                                      true_array: np.ndarray = None):
        method_name = 'postprocess_text_segmentation'
        try:
            def add_tags_to_word(word: str, tag_: str):
                if tag_:
                    for t in tag_:
                        word = f"<{t[1:-1]}>{word}</{t[1:-1]}>"
                    return word
                else:
                    return f"<p1>{word}</p1>"

            def reformat_tags(y_array: np.ndarray, tag_list: list, sensitivity: float = 0.9):
                norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
                reformat_list = []
                for word_tag in norm_array:
                    if np.sum(word_tag) == 0:
                        reformat_list.append(None)
                    else:
                        mix_tag = []
                        for wt, wtag in enumerate(word_tag):
                            if wtag == 1:
                                mix_tag.append(tag_list[wt])
                        reformat_list.append(mix_tag)
                return reformat_list

            def text_colorization(text: str, label_array: np.ndarray, tag_list: list):
                text = text.split(" ")
                labels = reformat_tags(label_array, tag_list)
                colored_text = []
                for w, word in enumerate(text):
                    colored_text.append(add_tags_to_word(word, labels[w]))
                return ' '.join(colored_text)

            # TODO: пока исходим что для сегментации текста есть только один вход с текстом, если будут сложные модели
            #  на сегментацию текста на несколько входов то придется искать решения

            classes_names = {}
            dataset_tags = dataset_params.get("open_tags").split()
            colors = {}
            if options.classes_colors and return_mode == 'deploy':
                for i, name in enumerate(dataset_tags):
                    colors[name] = options.classes_colors[i].as_rgb_tuple()
                    classes_names[name] = options.classes_names[i]
            elif class_colors and return_mode == 'callback':
                for i, name in enumerate(dataset_tags):
                    colors[name] = class_colors[i]
                    classes_names[name] = options.classes_names[i]
            else:
                hsv_tuples = [(x / len(dataset_tags), 1., 1.) for x in range(len(dataset_tags))]
                gen_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                gen_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), gen_colors))
                for i, name in enumerate(dataset_tags):
                    colors[name] = gen_colors[i]
                    classes_names[name] = options.classes_names[i]

            if return_mode == 'deploy':
                initinal_text = dataframe.iat[example_id, 0]
                text_segmentation = text_colorization(
                    text=initinal_text, label_array=pred_array, tag_list=dataset_tags
                )

                data = [('<p1>', '<p1>', (200, 200, 200))]
                for tag in colors.keys():
                    data.append(
                        (tag, classes_names[tag], colors[tag])
                    )
                return initinal_text, text_segmentation, data

            if return_mode == 'callback':
                data = {"y_true": {}, "y_pred": {}, "tags_color": {}, "stat": {}}
                text_for_preparation = dataframe.iat[example_id, 0]
                true_text_segmentation = text_colorization(
                    text=text_for_preparation, label_array=true_array, tag_list=dataset_tags,
                )

                data["y_true"] = {
                    "type": "segmented_text",
                    "data": [{"title": "Текст", "value": true_text_segmentation, "color_mark": None}]
                }
                pred_text_segmentation = text_colorization(
                    text=text_for_preparation, label_array=pred_array, tag_list=dataset_tags,
                )
                data["y_pred"] = {
                    "type": "segmented_text",
                    "data": [{"title": "Текст", "value": pred_text_segmentation, "color_mark": None}]
                }
                colors_ = {}
                for key, val in colors.items():
                    colors_[key[1:-1]] = val
                data["tags_color"] = colors_

                if show_stat:
                    data["stat"] = {
                        "type": "str",
                        "data": []
                    }
                    y_true = np.array(true_array).astype('int')
                    y_pred = np.where(pred_array >= 0.9, 1., 0.)
                    count = 0
                    mean_val = 0
                    for idx, cls in enumerate(options.classes_names):
                        if np.sum(y_true[:, idx]) == 0 and np.sum(y_pred[:, idx]) == 0:
                            data["stat"]["data"].append({'title': cls, 'value': "-", 'color_mark': None})
                        elif np.sum(y_true[:, idx]) == 0:
                            data["stat"]["data"].append({'title': cls, 'value': "0.0%", 'color_mark': 'wrong'})
                            count += 1
                        else:
                            class_recall = np.sum(y_true[:, idx] * y_pred[:, idx]) * 100 / np.sum(y_true[:, idx])
                            data["stat"]["data"].append(
                                {
                                    'title': cls,
                                    'value': f"{np.round(class_recall, 1)} %",
                                    'color_mark': 'success' if class_recall >= 90 else 'wrong'
                                }
                            )
                            count += 1
                            mean_val += class_recall
                    if count and mean_val / count >= 90:
                        mean_color_mark = "success"
                        mean_stat = f"{round(mean_val / count, 2)}%"
                    elif count and mean_val / count < 90:
                        mean_color_mark = "wrong"
                        mean_stat = f"{round(mean_val / count, 2)}%"
                    else:
                        mean_color_mark = None
                        mean_stat = '-'
                    data["stat"]["data"].insert(
                        0, {'title': "Средняя точность", 'value': mean_stat, 'color_mark': mean_color_mark}
                    )
                return data
        except Exception as e:
            print_error(TextSegmentationCallback().name, method_name, e)

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
                example_idx = TextSegmentationCallback().prepare_example_idx_to_show(
                    array=postprocess_array,
                    true_array=true_array[:len(array)],
                    options=options,
                    output=output_id,
                    count=int(len(array) * DEPLOY_PRESET_PERCENT / 100),
                    choice_type=ExampleChoiceTypeChoice.best,
                )
                return_data[output_id] = {"color_map": None, "data": []}
                output_column = list(options.instructions.get(output_id).keys())[0]
                for idx in example_idx:
                    source, segment, colors = TextSegmentationCallback().postprocess_text_segmentation(
                        pred_array=postprocess_array[idx],
                        options=options.data.outputs[output_id],
                        dataframe=options.dataframe.get("val"),
                        example_id=idx,
                        dataset_params=options.instructions.get(output_id).get(output_column),
                        return_mode='deploy'
                    )
                    return_data[output_id]["data"].append(
                        {"source": source, "format": segment}
                    )
                return_data[output_id]["color_map"] = colors
            return return_data
        except Exception as e:
            print_error(TextSegmentationCallback().name, method_name, e)

    @staticmethod
    def dataset_balance(options, y_true, preset_path: str, class_colors) -> dict:
        method_name = 'dataset_balance'
        try:
            dataset_balance = {}
            for out in options.data.outputs.keys():
                dataset_balance[f"{out}"] = {
                    "presence_balance": {},
                    "percent_balance": {}
                }
                for data_type in ['train', 'val']:
                    classes_names = options.data.outputs.get(out).classes_names
                    classes = np.arange(options.data.outputs.get(out).num_classes)
                    class_count = {}
                    class_percent = {}
                    for cl in classes:
                        class_count[classes_names[cl]] = \
                            np.sum(y_true.get(data_type).get(f"{out}")[..., cl]).item()
                        class_percent[options.data.outputs.get(out).classes_names[cl]] = np.round(
                            np.sum(y_true.get(data_type).get(f"{out}")[..., cl]) * 100
                            / np.prod(y_true.get(data_type).get(f"{out}")[..., cl].shape)).item()
                    dataset_balance[f"{out}"]["presence_balance"][data_type] = class_count
                    dataset_balance[f"{out}"]["percent_balance"][data_type] = class_percent
            return dataset_balance
        except Exception as e:
            print_error(TextSegmentationCallback().name, method_name, e)

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors, raw_y_pred) -> dict:
        method_name = 'intermediate_result_request'
        try:
            return_data = {}
            if interactive_config.intermediate_result.show_results:
                for idx in range(interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': {},
                        'statistic_values': {}
                    }
                    if not len(options.data.outputs.keys()) == 1:
                        for inp in options.data.inputs.keys():
                            data = ImageSegmentationCallback.postprocess_initial_source(
                                options=options,
                                input_id=inp,
                                save_id=idx + 1,
                                example_id=example_idx[idx],
                                dataset_path=dataset_path,
                                preset_path=preset_path,
                                return_mode='callback'
                            )
                            return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                                'type': 'text', 'data': data,
                            }
                    for out in options.data.outputs.keys():
                        output_col = list(options.instructions.get(out).keys())[0]
                        data = TextSegmentationCallback().postprocess_text_segmentation(
                            pred_array=y_pred.get(f'{out}')[example_idx[idx]],
                            true_array=y_true.get('val').get(f'{out}')[example_idx[idx]],
                            options=options.data.outputs.get(out),
                            dataframe=options.dataframe.get('val'),
                            example_id=example_idx[idx],
                            dataset_params=options.instructions.get(out).get(output_col),
                            return_mode='callback',
                            class_colors=class_colors,
                            show_stat=interactive_config.intermediate_result.show_statistic
                        )
                        if data.get('y_true'):
                            return_data[f"{idx + 1}"]['true_value'][f"Выходной слой «{out}»"] = data.get('y_true')
                        return_data[f"{idx + 1}"]['predict_value'][f"Выходной слой «{out}»"] = data.get('y_pred')

                        return_data[f"{idx + 1}"]['tags_color'][f"Выходной слой «{out}»"] = data.get('tags_color')

                        if data.get('stat'):
                            return_data[f"{idx + 1}"]['statistic_values'][f"Выходной слой «{out}»"] = data.get('stat')
                        else:
                            return_data[f"{idx + 1}"]['statistic_values'] = {}
            return return_data
        except Exception as e:
            print_error(TextSegmentationCallback().name, method_name, e)

    @staticmethod
    def statistic_data_request(interactive_config, options, y_true, inverse_y_true,
                               y_pred, inverse_y_pred, raw_y_pred=None) -> list:
        method_name = 'statistic_data_request'
        try:
            return_data = []
            _id = 1
            for out in interactive_config.statistic_data.output_id:
                encoding = options.data.outputs.get(out).encoding
                if encoding == LayerEncodingChoice.ohe:
                    cm, cm_percent = get_confusion_matrix(
                        np.argmax(y_true.get("val").get(f"{out}"), axis=-1).reshape(
                            np.prod(np.argmax(y_true.get("val").get(f"{out}"), axis=-1).shape)).astype('int'),
                        np.argmax(y_pred.get(f'{out}'), axis=-1).reshape(
                            np.prod(np.argmax(y_pred.get(f'{out}'), axis=-1).shape)).astype('int'),
                        get_percent=True
                    )
                    return_data.append(
                        fill_heatmap_front_structure(
                            _id=_id,
                            _type="heatmap",
                            graph_name=f"Выходной слой «{out}» - Confusion matrix",
                            short_name=f"{out} - Confusion matrix",
                            x_label="Предсказание",
                            y_label="Истинное значение",
                            labels=options.data.outputs.get(out).classes_names,
                            data_array=cm,
                            data_percent_array=cm_percent,
                        )
                    )
                    _id += 1
                elif encoding == LayerEncodingChoice.multi:
                    report = get_classification_report(
                        y_true=y_true.get("val").get(f"{out}").reshape(
                            (np.prod(y_true.get("val").get(f"{out}").shape[:-1]),
                             y_true.get("val").get(f"{out}").shape[-1])
                        ),
                        y_pred=np.where(y_pred.get(f"{out}") >= 0.9, 1, 0).reshape(
                            (np.prod(y_pred.get(f"{out}").shape[:-1]), y_pred.get(f"{out}").shape[-1])
                        ),
                        labels=options.data.outputs.get(out).classes_names
                    )
                    return_data.append(
                        fill_table_front_structure(
                            _id=_id,
                            graph_name=f"Выходной слой «{out}» - Отчет по классам",
                            plot_data=report
                        )
                    )
                    _id += 1
                else:
                    pass
            return return_data
        except Exception as e:
            print_error(TextSegmentationCallback().name, method_name, e)

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config) -> list:
        method_name = 'balance_data_request'
        try:
            return_data = []
            _id = 0
            for out in options.data.outputs.keys():
                for class_type in dataset_balance.get(f"{out}").keys():
                    preset = {}
                    if class_type in ["presence_balance", "percent_balance"]:
                        for data_type in ['train', 'val']:
                            names, count = sort_dict(
                                dict_to_sort=dataset_balance.get(f"{out}").get(class_type).get(data_type),
                                mode=interactive_config.data_balance.sorted.name
                            )
                            preset[data_type] = fill_graph_front_structure(
                                _id=_id,
                                _type='histogram',
                                type_data=data_type,
                                graph_name=f"Выход {out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - "
                                           f"{'баланс присутсвия' if class_type == 'presence_balance' else 'процент пространства'}",
                                short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                           f"{'присутсвие' if class_type == 'presence_balance' else 'процент'}",
                                x_label="Название класса",
                                y_label="Значение",
                                plot_data=[fill_graph_plot_data(x=names, y=count)],
                            )
                            _id += 1
                    return_data.append(preset)
            return return_data
        except Exception as e:
            print_error(TextSegmentationCallback().name, method_name, e)
