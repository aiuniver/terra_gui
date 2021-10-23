import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

__version__ = 0.08

from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.settings import CALLBACK_CLASSIFICATION_TREASHOLD_VALUE


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


class BalancedDiceCoef(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_dice_coef', encoding='ohe', **kwargs):
        super(BalancedDiceCoef, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        self.encoding = encoding
        # pass

    def update_state(self, y_true, y_pred, smooth=1, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if self.encoding == 'ohe':
            y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
        elif self.encoding == 'multi':
            y_pred = tf.where(y_pred > CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100, 1., 0.)
        else:
            pass

        balanced_dice = tf.convert_to_tensor(0., dtype=tf.float32)
        for i in range(y_true.shape[-1]):
            intersection = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
            union = K.sum(y_true[..., i:i + 1]) + K.sum(y_pred[..., i:i + 1])
            balanced_dice = tf.add(balanced_dice, tf.convert_to_tensor((2. * intersection + smooth) / (union + smooth)))
        self.dice = tf.convert_to_tensor(balanced_dice / y_true.shape[-1])

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedDiceCoef, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.dice

    def reset_state(self):
        self.dice: float = 0
        # pass


# class RecallPercent(tf.keras.metrics.Metric):
#
#     def __init__(self, name='recall_percent', **kwargs):
#         super(RecallPercent, self).__init__(name=name, **kwargs)
#         self.recall: float = 0
#         # pass
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
#         recall = K.sum(y_true * y_pred)
#         total = K.sum(y_true)
#         self.recall = tf.convert_to_tensor(recall * 100 / total)
#
#     def get_config(self):
#         """
#         Returns the serializable config of the metric.
#         """
#         config = super(RecallPercent, self).get_config()
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def result(self):
#         return self.recall
#
#     def reset_state(self):
#         self.recall: float = 0
#         # pass


class BalancedRecall(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_recall', **kwargs):
        super(BalancedRecall, self).__init__(name=name, **kwargs)
        self.recall: float = 0
        # pass

    def update_state(self, y_true, y_pred, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])

        if show_class:
            recall = K.sum(y_true * y_pred)
            total = K.sum(y_true)
            self.recall = tf.convert_to_tensor(recall * 100 / total)
        else:
            balanced_recall = tf.convert_to_tensor(0., dtype=tf.float32)
            for i in range(y_true.shape[-1]):
                recall = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
                total = K.sum(y_true[..., i:i + 1])
                if total == tf.convert_to_tensor(0.):
                    balanced_recall = tf.add(balanced_recall, tf.convert_to_tensor(0.))
                else:
                    balanced_recall = tf.add(balanced_recall, tf.convert_to_tensor(recall * 100 / total))
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


# class PrecisionPercent(tf.keras.metrics.Metric):
#
#     def __init__(self, name='precision_percent', **kwargs):
#         super(PrecisionPercent, self).__init__(name=name, **kwargs)
#         self.precision: float = 0
#         # pass
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
#         true_guess = K.sum(y_true * y_pred)
#         total = K.sum(y_pred)
#         self.precision = tf.convert_to_tensor(true_guess * 100 / total)
#
#     def get_config(self):
#         """
#         Returns the serializable config of the metric.
#         """
#         config = super(PrecisionPercent, self).get_config()
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def result(self):
#         return self.precision
#
#     def reset_state(self):
#         self.precision: float = 0
#         # pass


class BalancedPrecision(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_precision', **kwargs):
        super(BalancedPrecision, self).__init__(name=name, **kwargs)
        self.precision: float = 0
        # pass

    def update_state(self, y_true, y_pred, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])

        if show_class:
            true_guess = K.sum(y_true * y_pred)
            # total_true = K.sum(y_true)
            total_pred = K.sum(y_pred)
            self.precision = tf.convert_to_tensor(true_guess * 100 / total_pred)
        else:
            balanced_precision = tf.convert_to_tensor(0., dtype=tf.float32)
            for i in range(y_true.shape[-1]):
                true_guess = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
                total = K.sum(y_pred[..., i:i + 1])
                if total == tf.convert_to_tensor(0.):
                    balanced_precision = tf.add(balanced_precision, tf.convert_to_tensor(0.))
                else:
                    balanced_precision = tf.add(balanced_precision, tf.convert_to_tensor(true_guess * 100 / total))
            self.precision = tf.convert_to_tensor(balanced_precision / y_true.shape[-1])

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedPrecision, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.precision

    def reset_state(self):
        self.precision: float = 0
        # pass


class FScore(tf.keras.metrics.Metric):

    def __init__(self, name='f_score', **kwargs):
        super(FScore, self).__init__(name=name, **kwargs)
        self.score: float = 0
        # pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
        true_guess = K.sum(y_true * y_pred)
        total_true = K.sum(y_true)
        total_pred = K.sum(y_pred)
        recall = tf.convert_to_tensor(true_guess / total_true)
        precision = tf.convert_to_tensor(true_guess / total_pred)
        self.score = tf.convert_to_tensor(2 * precision * recall * 100 / (precision + recall))

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(FScore, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.score

    def reset_state(self):
        self.score: float = 0
        # pass


class BalancedFScore(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_f_score', **kwargs):
        super(BalancedFScore, self).__init__(name=name, **kwargs)
        self.score: float = 0
        # pass

    def update_state(self, y_true, y_pred, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])

        if show_class:
            true_guess = K.sum(y_true * y_pred)
            total_true = K.sum(y_true)
            total_pred = K.sum(y_pred)
            recall = tf.convert_to_tensor(true_guess / total_true)
            precision = tf.convert_to_tensor(true_guess / total_pred)
            self.score = tf.convert_to_tensor(2 * precision * recall * 100 / (precision + recall))
        else:
            balanced_score = tf.convert_to_tensor(0., dtype=tf.float32)
            for i in range(y_true.shape[-1]):
                true_guess = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
                total_true = K.sum(y_true[..., i:i + 1])
                total_pred = K.sum(y_pred[..., i:i + 1])
                if total_true == tf.convert_to_tensor(0.) or total_pred == tf.convert_to_tensor(0.):
                    balanced_score = tf.add(balanced_score, tf.convert_to_tensor(0.))
                else:
                    recall = tf.convert_to_tensor(true_guess / total_true)
                    precision = tf.convert_to_tensor(true_guess / total_pred)
                    balanced_score = tf.convert_to_tensor(2 * precision * recall * 100 / (precision + recall))
            self.score = balanced_score
            # self.score = tf.convert_to_tensor(balanced_score / y_true.shape[-1])

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedFScore, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.score

    def reset_state(self):
        self.score: float = 0
        # pass

class UnscaledMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name='unscaled_mae', **kwargs):
        super(UnscaledMAE, self).__init__(name=name, **kwargs)

    @staticmethod
    def unscale_result(mae_result, output: int, dataset: CreatePreprocessing):
        result = np.expand_dims(mae_result, axis=-1)
        preset = {output: {}}
        preprocess_dict = dataset.preprocessing.get(output)
        for i, col in enumerate(preprocess_dict.keys()):
            preset[output][col] = result[:, i:i + 1]
            break
        unscale = np.array(list(dataset.inverse_data(preset)[output].values()))
        try:
            return unscale.item()
        except ValueError:
            return unscale.squeeze().tolist()
