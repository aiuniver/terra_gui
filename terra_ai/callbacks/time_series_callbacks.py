import numpy as np


class TimeseriesCallback:
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

