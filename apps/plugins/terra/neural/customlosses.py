import tensorflow as tf
from tensorflow.keras import backend

__version__ = 0.07


class DiceCoefficient(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coef', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
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
        intersection = backend.sum(y_true * y_pred, axis=axis)
        union = backend.sum(y_true, axis=axis) + backend.sum(y_pred, axis=axis)
        self.dice = backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = super(DiceCoefficient, self).get_config()
        return config
    #
    # def get_config(self):
    #     config = {'n_layers': self.n_layers, 'filters': self.filters}
    #     base_config = super(resblock, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.dice

    def reset_states(self):
        self.dice: float = 0
        # pass


