import numpy as np

from terra_ai.data.datasets.extra import DatasetGroupChoice


class ImageClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
        if options.data.group == DatasetGroupChoice.keras:
            x_val = options.X.get("val")
        return x_val, inverse_x_val


class TextClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val


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


class AudioClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val


class VideoClassificationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

