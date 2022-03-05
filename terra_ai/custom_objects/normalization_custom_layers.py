import tensorflow as tf
import tensorflow
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self, axis=None, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, name='gamma', initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer, constraint=self.gamma_constraint
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape, name='beta', initializer=self.beta_initializer,
                regularizer=self.beta_regularizer, constraint=self.beta_constraint
            )
        self.built = True

    def call(self, inputs, training=True, **kwargs):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return input_shape


class Mish(Layer):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        - X_input = Input(input_shape)
        - X = Mish()(X_input)
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return inputs * tensorflow.math.tanh(tensorflow.math.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return input_shape


class RGBNormalization(Layer):

    def __init__(self, use_div2k_mean=False, denormalize=False, half_range_normalization=True, **kwargs):

        super(RGBNormalization, self).__init__(**kwargs)
        self.use_div2k_mean = use_div2k_mean
        self.denormalize = denormalize
        self.half_range_normalization = half_range_normalization

    def call(self, input_, training=False, **kwargs):
        DIV2K_RGB_MEAN = tf.convert_to_tensor([0.4488, 0.4371, 0.4040]) * 255
        # DIV2K_RGB_MEAN = tf.zeros_like(input_)
        # for batch in range(input_.shape[0]):
        #     DIV2K_RGB_MEAN[batch] = tf.convert_to_tensor([0.4488, 0.4371, 0.4040]) * 255

        if self.denormalize:
            if self.half_range_normalization and self.use_div2k_mean:
                output = input_ * 127.5 + DIV2K_RGB_MEAN
            elif self.half_range_normalization:
                output = (input_ + 1) * 127.5
            else:
                output = input_ * 255.
        else:
            if self.half_range_normalization and self.use_div2k_mean:
                output = (input_ - DIV2K_RGB_MEAN) / 127.5
            elif self.half_range_normalization:
                output = input_ / 127.5 - 1
            else:
                output = input_ / 255.
        return output

    def get_config(self):
        config = {
            'use_div2k_mean': self.filters,
            'denormalize': self.kernel_size,
            'half_range_normalization': self.kernel_initializer,
        }
        base_config = super(RGBNormalization, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == "__main__":
    pass
