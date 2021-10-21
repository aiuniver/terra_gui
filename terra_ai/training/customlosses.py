import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

__version__ = 0.08

from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.datasets.preprocessing import CreatePreprocessing


class DiceCoef(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coef', **kwargs):
        super(DiceCoef, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        # pass

    def update_state(self, y_true, y_pred, smooth=1, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # по размеру тензора определяем пришли маски по изображениям или по тексту
        shape = y_true.shape
        if len(shape) == 3:
            axis = [1, 2]
        elif len(shape) == 4:
            axis = [1, 2, 3]
        else:
            axis = [1]
        intersection = K.sum(y_true * y_pred, axis=axis)
        union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
        self.dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(DiceCoef, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.dice

    def reset_state(self):
        self.dice: float = 0
        # pass


class RecallPercent(tf.keras.metrics.Metric):

    def __init__(self, name='recall_percent', **kwargs):
        super(RecallPercent, self).__init__(name=name, **kwargs)
        self.recall: float = 0
        # pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
        recall = K.sum(y_true * y_pred)
        total = K.sum(y_true)
        self.recall = tf.convert_to_tensor(recall * 100 / total)

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(RecallPercent, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.recall

    def reset_state(self):
        self.recall: float = 0
        # pass


class BalancedRecall(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_recall', **kwargs):
        super(BalancedRecall, self).__init__(name=name, **kwargs)
        self.recall: float = 0
        # pass

    def update_state(self, y_true, y_pred, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
        # recall = K.sum(y_true * y_pred)
        # total = K.sum(y_true)
        # self.recall = tf.convert_to_tensor(recall * 100 / total)

        if show_class:
            recall = K.sum(y_true * y_pred)
            total = K.sum(y_true)
            self.recall = tf.convert_to_tensor(recall * 100 / total)
            # self.recall = self.recall if self.recall else tf.convert_to_tensor(0., dtype=tf.float32)
        else:
            # for i in range(y_true.shape[-1]):

            balanced_recall = tf.convert_to_tensor(0., dtype=tf.float32)
            for i in range(y_true.shape[-1]):
                recall = K.sum(y_true[..., i:i+1] * y_pred[..., i:i+1])
                total = K.sum(y_true[..., i:i+1])
                if total == tf.convert_to_tensor(0.):
                    balanced_recall = tf.add(balanced_recall, tf.convert_to_tensor(0.))
                else:
                    balanced_recall = tf.add(balanced_recall, tf.convert_to_tensor(recall * 100 / total))
                # balanced_recall = tf.add(balanced_recall, tf.convert_to_tensor(recall * 100 / total))
            # print('\n', balanced_recall, tf.cast(y_true.shape[-1], tf.float32))
            # if not balanced_recall:
            #     balanced_recall = tf.convert_to_tensor(0., dtype=tf.float32)
                # print('self.recall', i, total)
            self.recall = tf.convert_to_tensor(balanced_recall / y_true.shape[-1])

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedRecall, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.recall

    def reset_state(self):
        self.recall: float = 0
        # pass


class UnscaledMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name='unscaled_mae', **kwargs):
        super(UnscaledMAE, self).__init__(name=name, **kwargs)

    @staticmethod
    def unscale_result(mae_result, output: int, dataset: CreatePreprocessing):
        result = np.array(mae_result).reshape(-1, 1)
        preset = {output: {}}
        preprocess_dict = dataset.preprocessing.get(output)
        for i, col in enumerate(preprocess_dict.keys()):
            preset[output][col] = result[:, i:i + 1]
        unscale = np.array(list(dataset.inverse_data(preset)[output].values()))
        try:
            return unscale.item()
        except ValueError:
            return unscale.squeeze().tolist()

