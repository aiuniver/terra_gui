import numpy as np
import pandas as pd

from terra_ai.callbacks.utils import sort_dict, get_y_true, get_distribution_histogram, get_correlation_matrix
from terra_ai.data.datasets.extra import LayerInputTypeChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.settings import CALLBACK_REGRESSION_TREASHOLD_VALUE, DEPLOY_PRESET_PERCENT


class DataframeRegressionCallback:
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
    def get_y_true(options):
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

                preprocess_dict = options.preprocessing.preprocessing.get(out)
                inverse_y = np.zeros_like(y_true.get(data_type).get(f"{out}")[:, 0:1])
                for i, column in enumerate(preprocess_dict.keys()):
                    if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                        _options = {int(out): {column: y_true.get(data_type).get(f"{out}")[:, i:i + 1]}}
                        inverse_col = options.preprocessing.inverse_data(_options).get(out).get(column)
                    else:
                        inverse_col = y_true.get(data_type).get(f"{out}")[:, i:i + 1]
                    inverse_y = np.concatenate([inverse_y, inverse_col], axis=-1)
                inverse_y_true[data_type][f"{out}"] = inverse_y[:, 1:]
        return y_true, inverse_y_true

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
                array=postprocess_array[:len(array)],
                true_array=true_array[:len(array)],
                count=int(len(array) * DEPLOY_PRESET_PERCENT / 100)
            )
            return_data[output_id] = {
                'preset': [],
                'label': []
            }
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
                    channel_inverse_col.append(str(inverse_col[0]))
                return_data[output_id]['label'].append(channel_inverse_col)
        return return_data

    @staticmethod
    def dataset_balance(options) -> dict:
        dataset_balance = {}
        for out in options.data.outputs.keys():
            dataset_balance[f"{out}"] = {'histogram': {}, 'correlation': {}}
            for data_type in ['train', 'val']:
                dataset_balance[f"{out}"]['histogram'][data_type] = {}
                for column in list(options.dataframe.get('train').columns):
                    column_id = int(column.split("_")[0])
                    column_task = options.data.columns.get(column_id).get(column).task
                    column_data = list(options.dataframe.get(data_type)[column])
                    if column_task == LayerInputTypeChoice.Text:
                        continue
                    elif column_task == LayerInputTypeChoice.Classification:
                        x, y = get_distribution_histogram(column_data, categorical=True)
                        hist_type = "histogram"
                    else:
                        x, y = get_distribution_histogram(column_data, categorical=False)
                        hist_type = "bar"
                    dataset_balance[f"{out}"]['histogram'][data_type][column] = {
                        "name": column.split("_", 1)[-1],
                        "type": hist_type,
                        "x": x,
                        "y": y
                    }
            for data_type in ['train', 'val']:
                labels, matrix = get_correlation_matrix(pd.DataFrame(options.dataframe.get(data_type)))
                dataset_balance[f"{out}"]['correlation'][data_type] = {
                    "labels": labels,
                    "matrix": matrix
                }
        return dataset_balance


def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, count: int,
                                choice_type: str = "best", seed_idx: list = None) -> dict:
    example_idx = []
    if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
        delta = np.abs(true_array - array) * 100 / true_array
        while len(delta.shape) != 1:
            delta = np.mean(delta, axis=-1)
        delta_dict = dict(zip(np.arange(0, len(delta)), delta))
        if choice_type == ExampleChoiceTypeChoice.best:
            example_idx, _ = sort_dict(delta_dict, mode=BalanceSortedChoice.ascending)
            example_idx = example_idx[:count]
        if choice_type == ExampleChoiceTypeChoice.worst:
            example_idx, _ = sort_dict(delta_dict, mode=BalanceSortedChoice.descending)
            example_idx = example_idx[:count]

    elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
        example_idx = seed_idx[:count]

    elif choice_type == ExampleChoiceTypeChoice.random:
        delta = np.abs(true_array - array) * 100 / true_array
        while len(delta.shape) != 1:
            delta = np.mean(delta, axis=-1)
        true_id = []
        false_id = []
        for i, ex in enumerate(delta):
            if ex >= CALLBACK_REGRESSION_TREASHOLD_VALUE:
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


def postprocess_regression(column_names: list, inverse_y_true: np.ndarray, inverse_y_pred: np.ndarray,
                           show_stat: bool = False, return_mode='deploy'):
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

