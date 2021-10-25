import copy
import os

import numpy as np
from PIL import Image
from pydub import AudioSegment
from tensorflow.keras.preprocessing import image

from terra_ai.callbacks.utils import sort_dict, fill_graph_front_structure, fill_graph_plot_data, get_y_true
from terra_ai.data.datasets.dataset import DatasetOutputsData
from terra_ai.data.datasets.extra import DatasetGroupChoice, LayerInputTypeChoice, LayerEncodingChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.datasets.preparing import PrepareDataset
import moviepy.editor as moviepy_editor

from terra_ai.settings import MAX_GRAPH_LENGTH, DEPLOY_PRESET_PERCENT


class ImageClassificationCallback:
    def __init__(self, options: PrepareDataset):
        self.options = options
        pass

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
        if options.data.group == DatasetGroupChoice.keras:
            x_val = options.X.get("val")
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_initial_source(options: PrepareDataset, input_id: int, example_id: int, dataset_path: str,
                                   preset_path: str, save_id: int = None, x_array=None, return_mode='deploy'):
        column_idx = []
        if options.data.group != DatasetGroupChoice.keras:
            for inp in options.data.inputs.keys():
                for column_name in options.dataframe.get('val').columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))
            initial_file_path = os.path.join(
                dataset_path, options.dataframe.get('val').iat[example_id, column_idx[0]]
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
            img = image.array_to_img(x_array[example_id])
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

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        x_array, inverse_x_array = ImageClassificationCallback.get_x_array(options)
        return_data = {}

        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )

            return_data[output_id] = []
            _id = 1
            for idx in example_idx:
                input_id = list(options.data.inputs.keys())[0]
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
                actual_value, predict_values = postprocess_classification(
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


class TextClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array():
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_initial_source(options, example_id: int, return_mode='deploy'):
        column_idx = []
        data = []
        data_type = ""
        source = ""

        for column in column_idx:
            source = options.dataframe.get('val').iat[example_id, column]
            if return_mode == 'deploy':
                break
            if return_mode == 'callback':
                data_type = LayerInputTypeChoice.Text.name
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
            return data, data_type.lower()

    @staticmethod
    def postprocess_deploy(array, options) -> dict:
        return_data = {}

        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )

            return_data[output_id] = []
            _id = 1
            for idx in example_idx:
                source = TextClassificationCallback.postprocess_initial_source(
                    options=options,
                    example_id=idx,
                    return_mode='deploy'
                )
                actual_value, predict_values = postprocess_classification(
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


class DataframeClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
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

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, return_mode='deploy'):
        data = []
        data_type = "str"
        source = []
        for col_name in options.data.columns.get(input_id).keys():
            value = options.dataframe.get('val')[col_name].to_list()[example_id]
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
            return data, data_type.lower()

    @staticmethod
    def postprocess_deploy(array, options) -> dict:
        return_data = {}
        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )
            return_data[output_id] = []
            _id = 1
            for idx in example_idx:
                input_id = list(options.data.inputs.keys())[0]
                source = DataframeClassificationCallback.postprocess_initial_source(
                    options=options,
                    input_id=input_id,
                    example_id=idx,
                    return_mode='deploy'
                )
                actual_value, predict_values = postprocess_classification(
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


class AudioClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array():
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_initial_source(options: PrepareDataset, input_id: int, example_id: int, dataset_path: str,
                                   preset_path: str, save_id: int = None, return_mode='deploy'):
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

        data = []
        data_type = ""
        source = os.path.join(preset_path, f"initial_data_audio_{save_id}_input_{input_id}.webm")
        AudioSegment.from_file(initial_file_path).export(source, format="webm")
        if return_mode == 'callback':
            data_type = LayerInputTypeChoice.Audio.name
            data = [
                {
                    "title": "Аудио",
                    "value": source,
                    "color_mark": None
                }
            ]
        if return_mode == 'deploy':
            return source
        if return_mode == 'callback':
            return data, data_type.lower()

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        return_data = {}

        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
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
                actual_value, predict_values = postprocess_classification(
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


class VideoClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array():
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, dataset_path: str, preset_path: str,
                                   save_id: int = None, return_mode='deploy'):
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

        clip = moviepy_editor.VideoFileClip(initial_file_path)
        source = os.path.join(preset_path, f"initial_data_video_{save_id}_input_{input_id}.webm")
        clip.write_videofile(source)

        if return_mode == 'deploy':
            return source
        if return_mode == 'callback':
            data_type = LayerInputTypeChoice.Video.name
            data = [
                {
                    "title": "Видео",
                    "value": source,
                    "color_mark": None
                }
            ]
            return data, data_type.lower()

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        return_data = {}

        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
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
                actual_value, predict_values = postprocess_classification(
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


class TimeseriesTrendCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
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

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, inverse_x_array=None,
                                   return_mode='deploy'):
        column_idx = []
        for inp in options.data.inputs.keys():
            for column_name in options.dataframe.get('val').columns:
                if column_name.split('_')[0] == f"{inp}":
                    column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))

        source = ""
        graphics_data = []
        names = ""
        multi = False
        if return_mode == 'callback':
            for i, channel in enumerate(options.data.columns.get(input_id).keys()):
                multi = True if i > 0 else False
                names += f"«{channel.split('_', 1)[-1]}», "
                length = len(inverse_x_array) if len(inverse_x_array) < MAX_GRAPH_LENGTH else MAX_GRAPH_LENGTH
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
                                x=np.arange(inverse_x_array[example_id].shape[-2]).astype('int').tolist()[
                                  -length:],
                                y=inverse_x_array[example_id][:, i].astype('float').tolist()[-length:]
                            )
                        ],
                    )
                )
            data_type = "graphic"
            data = [
                {
                    "title": f"График{'и' if multi else ''} по канал{'ам' if multi else 'у'} {names[:-2]}",
                    "value": graphics_data,
                    "color_mark": None
                }
            ]
            return data, data_type.lower()

        if return_mode == 'deploy':
            return source

    @staticmethod
    def postprocess_results(array, options, save_path: str = "") -> dict:
        return_data = {}
        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )
            return_data[output_id] = {}
            # TODO: считаетм что инпут один
            input_id = list(options.data.inputs.keys())[0]
            inp_col_id = []
            for j, out_col in enumerate(options.data.columns.get(output_id).keys()):
                for k, inp_col in enumerate(options.data.columns.get(input_id).keys()):
                    if out_col.split('_', 1)[-1] == inp_col.split('_', 1)[-1]:
                        inp_col_id.append((k, inp_col, j, out_col))
                        break
            preprocess = options.preprocessing.preprocessing.get(output_id)
            for channel in inp_col_id:
                return_data[output_id][channel[3]] = []
                for idx in example_idx:
                    if type(preprocess.get(channel[3])).__name__ in ['StandardScaler', 'MinMaxScaler']:
                        inp_options = {int(output_id): {
                            channel[3]: options.X.get('val').get(f"{input_id}")[idx, channel[0]:channel[0] + 1]}
                        }
                        inverse_true = options.preprocessing.inverse_data(inp_options).get(output_id).get(
                            channel[3])
                        inverse_true = inverse_true.squeeze().astype('float').tolist()
                    else:
                        inverse_true = options.X.get('val').get(f"{input_id}")[
                                       idx, channel[0]:channel[0] + 1].squeeze().astype('float').tolist()
                    actual_value, predict_values = postprocess_classification(
                        predict_array=np.expand_dims(postprocess_array[idx], axis=0),
                        true_array=true_array[idx],
                        options=options.data.outputs[output_id],
                        return_mode='deploy'
                    )
                    button_save_path = os.path.join(
                        save_path, f"ts_trend_button_channel_{channel[2]}_image_{idx}.jpg")
                    return_data[output_id][channel[3]].append(
                        {
                            "button_link": button_save_path,
                            "data": [inverse_true, predict_values]
                        }
                    )

        return return_data


def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                choice_type: str = "best", seed_idx: list = None) -> dict:
    example_idx = []
    encoding = options.data.outputs.get(output).encoding
    if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
        if array.shape[-1] == true_array.shape[-1] and encoding == LayerEncodingChoice.ohe and true_array.shape[-1] > 1:
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
            key = np.random.choice(list(class_idx.keys()))
            if choice_type == ExampleChoiceTypeChoice.best:
                example_idx.append(class_idx[key][-1])
                class_idx[key].pop(-1)
            if choice_type == ExampleChoiceTypeChoice.worst:
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


def postprocess_classification(predict_array: np.ndarray, true_array: np.ndarray, options: DatasetOutputsData,
                               show_stat: bool = False, return_mode='deploy'):
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


