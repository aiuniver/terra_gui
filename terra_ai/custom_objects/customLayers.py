import logging

import numpy as np
import tensorflow as tf
import tensorflow
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.python.layers.base import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow import cast
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import InputSpec

terra_custom_layers = {
    "InstanceNormalization": "customLayers",
    "VAEBlock": "customLayers",
    "YOLOResBlock": "customLayers",
    "YOLOv3ResBlock": "customLayers",
    "YOLOConvBlock": "customLayers",
    "Mish": "customLayers",
    # "DarkNetBatchNormalization": "customLayers",
    # "DarkNetConvolutional": "customLayers",
    # "DarkNetResBlock": "customLayers",
    # "DarkNetUpsample": "customLayers",
    "CONVBlock": "customLayers",
    "PSPBlock2D": "customLayers",
    "UNETBlock2D": "customLayers",
    "UNETBlock1D": "customLayers",
    "UNETBlock3D": "customLayers",
    "PSPBlock1D": "customLayers",
    "PretrainedYOLO": "customLayers",
    # "DarknetBatchNormalization": "custom_objects/pretrained_yolo"
    "OnlyYolo": "customLayers",
    "ConditionalMergeLayer": "customLayers",
    "ResnetBlock2D": "customLayers",
    "Transformer": "customLayers",
    "PretrainedBERT": "customLayers",
}


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


class VAEBlock(Layer):
    """
    Custom Layer VAEBlock
    Keras Layer to grab a random sample from a distribution (by multiplication)
    Computes "(normal)*stddev + mean" for the vae sampling operation
    (written for tf backend)
    Additionally,
        Applies regularization to the latent space representation.
        Can perform standard regularization or B-VAE regularization.
    call:
        pass in mean then stddev layers to sample from the distribution
        ex.
            sample = SampleLayer('bvae', 16)([mean, stddev])
    """

    def __init__(self, latent_size=32, latent_regularizer='vae', beta=5.,
                 capacity=128., random_sample=True, roll_up=True, **kwargs):
        """
        args:
        ------
        latent_regularizer : str
            Either 'bvae', 'vae', or None
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer,
            (Unused if 'bvae' not selected)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        random_sample : bool
            whether or not to use random sampling when selecting from
                distribution.
            if false, the latent vector equals the mean, essentially turning
                this into a standard autoencoder.
        latent_size : int
        roll_up: bool
        ------
        ex.
            sample = VAEBlock(latent_regularizer='bvae', beta=16,
                              latent_size=32)(x)
        """
        super(VAEBlock, self).__init__(**kwargs)
        # sampling
        self.reg = latent_regularizer
        self.beta = beta
        self.capacity = capacity
        self.random = random_sample
        # variational encoder
        self.latent_size = latent_size
        self.roll_up = roll_up
        self.conv_mean = layers.Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                                       padding='same')
        self.gla_mean = layers.GlobalAveragePooling2D()
        self.conv_stddev = layers.Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                                         padding='same')
        self.gla_stddev = layers.GlobalAveragePooling2D()

        self.conv1d_mean = layers.Conv1D(filters=self.latent_size, kernel_size=1,
                                         padding='same')
        self.gla1d_mean = layers.GlobalAveragePooling1D()
        self.conv1d_stddev = layers.Conv1D(filters=self.latent_size, kernel_size=1,
                                           padding='same')
        self.gla1d_stddev = layers.GlobalAveragePooling1D()

        self.inter_dense = layers.Dense(8 * self.latent_size, activation='relu')
        self.dense_mean = layers.Dense(self.latent_size)
        self.dense_stddev = layers.Dense(self.latent_size)

    def call(self, inputs, training=True, **kwargs):
        # variational encoder output (distributions)
        if K.ndim(inputs) == 4:
            mean = self.conv_mean(inputs)
            stddev = self.conv_stddev(inputs)
            if self.roll_up:
                mean = self.gla_mean(mean)
                stddev = self.gla_stddev(stddev)

        elif K.ndim(inputs) == 3:
            mean = self.conv1d_mean(inputs)
            stddev = self.conv1d_stddev(inputs)
            if self.roll_up:
                mean = self.gla1d_mean(mean)
                stddev = self.gla1d_stddev(stddev)

        elif K.ndim(inputs) == 2:
            inter = self.inter_dense(inputs)
            mean = self.dense_mean(inter)
            stddev = self.dense_stddev(inter)
        else:
            raise Exception(
                'input shape VAEBlock is not a vector [batchSize, intermediate_dim] or [batchSize, width, heigth, ch] \
                or [batchSize, steps, input_dim')
        if self.reg:
            # kl divergence:
            latent_loss = K.mean(-0.5 * K.sum(1 + stddev
                                              - K.square(mean)
                                              - K.exp(stddev), axis=-1))
            if self.reg == 'bvae':
                # use beta to force less usage of vector space:
                # also try to use <capacity> dimensions of the space:
                latent_loss = self.beta * K.abs(latent_loss - self.capacity / self.latent_size)
            self.add_loss(latent_loss)

        epsilon = K.random_normal(shape=K.shape(mean),
                                  mean=0., stddev=1.)

        if self.random:
            # 'reparameterization trick':
            return mean + K.exp(stddev / 2) * epsilon
        else:  # do not perform random sampling, simply grab the impulse value
            return mean + 0 * stddev  # Keras needs the *0 so the gradient is not None

    # def compute_output_shape(self, input_shape):
    #     return tf.shape(input_shape)[0]

    def get_config(self):
        config = {
            'latent_regularizer': self.reg,
            'beta': self.beta,
            'capacity': self.capacity,
            'randomSample': self.random,
            'latent_size': self.latent_size,
            'roll_up': self.roll_up,
        }
        base_config = super(VAEBlock, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class YOLOResBlock(Layer):
    def __init__(self, mode="YOLOv3", filters=32, num_resblocks=1, activation='LeakyReLU', use_bias=False,
                 include_head=True, include_add=True, all_narrow=False, **kwargs):
        super(YOLOResBlock, self).__init__(**kwargs)
        self.mode = mode
        self.all_narrow = all_narrow
        self.filters = filters
        self.num_resblocks = num_resblocks
        self.activation = activation
        self.activation_choice = [tensorflow.keras.layers.LeakyReLU,
                                  Mish,
                                  tensorflow.keras.layers.Activation]
        if self.activation == 'LeakyReLU':
            self.activ_state = (0, {'alpha': 0.1})
        if self.activation == 'Mish':
            self.activ_state = (1, {})
        if self.activation == 'Swish':
            self.activ_state = (2, {'activation': 'swish'})
        self.use_bias = use_bias
        self.include_head = include_head
        self.include_add = include_add

        self.kwargs = {'use_bias': self.use_bias, 'activation': 'linear'}
        if self.mode == "YOLOv3":
            self.kwargs["kernel_regularizer"] = tensorflow.keras.regularizers.l2(5e-4)
            self.kwargs["kernel_initializer"] = tensorflow.keras.initializers.RandomNormal(stddev=0.01)
            self.kwargs["bias_initializer"] = tensorflow.keras.initializers.Constant(value=0)
        if self.mode == "YOLOv4":
            self.kwargs["kernel_initializer"] = tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        if self.mode == "YOLOv5":
            self.kwargs["kernel_initializer"] = tensorflow.keras.initializers.RandomNormal(stddev=0.01)
            self.kwargs["kernel_regularizer"] = tensorflow.keras.regularizers.l2(5e-4)

        if self.include_head:
            self.zero2d = tensorflow.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))
            self.conv_start = tensorflow.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3),
                                                             strides=(2, 2), padding='valid', **self.kwargs)
            self.bn_start = tensorflow.keras.layers.BatchNormalization(momentum=0.03 if self.mode == "YOLOv5" else 0.99)
            self.activation_start = self.activation_choice[self.activ_state[0]](**self.activ_state[1])

        if self.mode == "YOLOv4" or self.mode == "YOLOv5":
            self.preconv_1 = tensorflow.keras.layers.Conv2D(
                filters=self.filters // 2 if self.all_narrow else self.filters, kernel_size=(1, 1),
                padding='same', **self.kwargs)
            self.prebn_1 = tensorflow.keras.layers.BatchNormalization(momentum=0.03 if self.mode == "YOLOv5" else 0.99)
            self.preactivation_1 = self.activation_choice[self.activ_state[0]](**self.activ_state[1])
            self.preconv_2 = tensorflow.keras.layers.Conv2D(
                filters=self.filters // 2 if self.all_narrow else self.filters,
                kernel_size=(3, 3) if self.mode == "YOLOv5" else (1, 1),
                padding='same', **self.kwargs)
            self.prebn_2 = tensorflow.keras.layers.BatchNormalization(momentum=0.03 if self.mode == "YOLOv5" else 0.99)
            self.preactivation_2 = self.activation_choice[self.activ_state[0]](**self.activ_state[1])

        for i in range(self.num_resblocks):
            setattr(self, f"conv_1_{i}",
                    tensorflow.keras.layers.Conv2D(filters=self.filters // 2, kernel_size=(1, 1),
                                                   padding='same', **self.kwargs))
            setattr(self, f"conv_2_{i}",
                    tensorflow.keras.layers.Conv2D(filters=self.filters // 2 if (
                            self.all_narrow and self.mode in ["YOLOv4", "YOLOv5"]) else self.filters,
                                                   kernel_size=(3, 3), padding='same', **self.kwargs))
            setattr(self, f"bn_1_{i}", tensorflow.keras.layers.BatchNormalization(
                momentum=0.03 if self.mode == "YOLOv5" else 0.99))
            setattr(self, f"bn_2_{i}", tensorflow.keras.layers.BatchNormalization(
                momentum=0.03 if self.mode == "YOLOv5" else 0.99))
            setattr(self, f"activ_1_{i}", self.activation_choice[self.activ_state[0]](**self.activ_state[1]))
            setattr(self, f"activ_2_{i}", self.activation_choice[self.activ_state[0]](**self.activ_state[1]))
            if self.include_add:
                setattr(self, f"add_{i}", tensorflow.keras.layers.Add())

        if self.mode == "YOLOv4":
            self.postconv_1 = tensorflow.keras.layers.Conv2D(
                filters=self.filters // 2 if self.all_narrow else self.filters, kernel_size=(1, 1),
                padding='same', **self.kwargs)
            self.postbn_1 = tensorflow.keras.layers.BatchNormalization(momentum=0.03 if self.mode == "YOLOv5" else 0.99)
            self.postactivation_1 = self.activation_choice[self.activ_state[0]](**self.activ_state[1])
        if self.mode == "YOLOv4" or self.mode == "YOLOv5":
            self.concatenate_1 = tensorflow.keras.layers.Concatenate()
            self.postconv_2 = tensorflow.keras.layers.Conv2D(
                filters=self.filters, kernel_size=(1, 1), padding='same', **self.kwargs)
            self.postbn_2 = tensorflow.keras.layers.BatchNormalization(momentum=0.03 if self.mode == "YOLOv5" else 0.99)
            self.postactivation_2 = self.activation_choice[self.activ_state[0]](**self.activ_state[1])

    def call(self, x, training=True, **kwargs):
        if self.include_head:
            x = self.zero2d(x)
            x = self.conv_start(x)
            x = self.bn_start(x)
            x = self.activation_start(x)
        if self.mode == "YOLOv4" or self.mode == "YOLOv5":
            x_concat = self.preconv_1(x)
            x_concat = self.prebn_1(x_concat)
            x_concat = self.preactivation_1(x_concat)
            x = self.preconv_2(x)
            x = self.prebn_2(x)
            x = self.preactivation_2(x)
        for i in range(self.num_resblocks):
            y = getattr(self, f"conv_1_{i}")(x)
            y = getattr(self, f"bn_1_{i}")(y)
            y = getattr(self, f"activ_1_{i}")(y)
            y = getattr(self, f"conv_2_{i}")(y)
            y = getattr(self, f"bn_2_{i}")(y)
            if self.include_add:
                y = getattr(self, f"activ_2_{i}")(y)
                x = getattr(self, f"add_{i}")([y, x])
            else:
                x = getattr(self, f"activ_2_{i}")(y)
        if self.mode == "YOLOv4":
            x = self.postconv_1(x)
            x = self.postbn_1(x)
            x = self.postactivation_1(x)
        if self.mode == "YOLOv4" or self.mode == "YOLOv5":
            x = self.concatenate_1([x, x_concat])
            x = self.postconv_2(x)
            x = self.postbn_2(x)
            x = self.postactivation_2(x)
        return x

    def get_config(self):
        config = {
            'mode': self.mode,
            'filters': self.filters,
            'num_resblocks': self.num_resblocks,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'include_head': self.include_head,
            'include_add': self.include_add,
            'all_narrow': self.all_narrow
        }
        base_config = super(YOLOResBlock, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class YOLOv3ResBlock(Layer):

    def __init__(self, filters=32, num_resblocks=1, use_bias=False, include_head=True, **kwargs):
        super(YOLOv3ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.num_resblocks = num_resblocks
        self.include_head = include_head
        self.use_bias = use_bias
        self.kwargs = {'use_bias': use_bias, 'activation': None,
                       "kernel_regularizer": tensorflow.keras.regularizers.l2(5e-4),
                       "kernel_initializer": tensorflow.keras.initializers.RandomNormal(stddev=0.01),
                       "bias_initializer": tensorflow.keras.initializers.Constant(value=0)}
        # self.kwargs.update(kwargs)
        if self.include_head:
            self.zero2d = tensorflow.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))
            self.conv_start = tensorflow.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3),
                                                             strides=(2, 2), padding='valid', **self.kwargs)
            self.bn_start = tensorflow.keras.layers.BatchNormalization(momentum=0.99)
            self.activation_start = tensorflow.keras.layers.LeakyReLU(**{'alpha': 0.1})

        for i in range(self.num_resblocks):
            setattr(self, f"conv_1_{i}",
                    tensorflow.keras.layers.Conv2D(filters=self.filters // 2, kernel_size=(1, 1),
                                                   padding='same', **self.kwargs))
            setattr(self, f"conv_2_{i}",
                    tensorflow.keras.layers.Conv2D(filters=self.filters,
                                                   kernel_size=(3, 3), padding='same', **self.kwargs))
            setattr(self, f"bn_1_{i}", tensorflow.keras.layers.BatchNormalization(momentum=0.99))
            setattr(self, f"bn_2_{i}", tensorflow.keras.layers.BatchNormalization(momentum=0.99))
            setattr(self, f"activ_1_{i}", tensorflow.keras.layers.LeakyReLU(**{'alpha': 0.1}))
            setattr(self, f"activ_2_{i}", tensorflow.keras.layers.LeakyReLU(**{'alpha': 0.1}))
            setattr(self, f"add_{i}", tensorflow.keras.layers.Add())

    def call(self, x, training=True, **kwargs):
        if self.include_head:
            x = self.zero2d(x)
            x = self.conv_start(x)
            x = self.bn_start(x)
            x = self.activation_start(x)
        for i in range(self.num_resblocks):
            y = getattr(self, f"conv_1_{i}")(x)
            y = getattr(self, f"bn_1_{i}")(y)
            y = getattr(self, f"activ_1_{i}")(y)
            y = getattr(self, f"conv_2_{i}")(y)
            y = getattr(self, f"bn_2_{i}")(y)
            y = getattr(self, f"activ_2_{i}")(y)
            x = getattr(self, f"add_{i}")([y, x])
        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'num_resblocks': self.num_resblocks,
            'use_bias': self.use_bias,
            'include_head': self.include_head,
        }
        base_config = super(YOLOv3ResBlock, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class YOLOConvBlock(Layer):
    """Unet block layer """

    def __init__(self, mode="YOLOv3", filters=32, num_conv=1, activation='LeakyReLU', use_bias=False,
                 first_conv_kernel=(1, 1), first_conv_strides=(1, 1), first_conv_padding='same',
                 include_bn_activation=True, **kwargs):
        super(YOLOConvBlock, self).__init__(**kwargs)
        self.mode = mode
        self.use_bias = use_bias
        self.strides = first_conv_strides
        self.kernel = first_conv_kernel
        self.padding = first_conv_padding
        self.include_bn_activation = include_bn_activation
        self.kwargs = {'activation': 'linear', 'use_bias': self.use_bias}
        if self.mode == "YOLOv3":
            self.kwargs["kernel_regularizer"] = tensorflow.keras.regularizers.l2(5e-4)
            self.kwargs["kernel_initializer"] = tensorflow.keras.initializers.RandomNormal(stddev=0.01)
            self.kwargs["bias_initializer"] = tensorflow.keras.initializers.Constant(value=0)
        if self.mode == "YOLOv4":
            self.kwargs["kernel_initializer"] = tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        if self.mode == "YOLOv5":
            self.kwargs["kernel_initializer"] = tensorflow.keras.initializers.RandomNormal(stddev=0.01)
            self.kwargs["kernel_regularizer"] = tensorflow.keras.regularizers.l2(5e-4)

        self.filters = filters
        self.num_conv = num_conv
        self.mode = mode
        self.activation = activation
        self.use_bias = use_bias
        self.strides = first_conv_strides
        self.kernel = first_conv_kernel
        self.padding = first_conv_padding

        for i in range(self.num_conv):
            if i == 0:
                setattr(self, f"conv_{i}", tensorflow.keras.layers.Conv2D(
                    filters=self.filters, kernel_size=self.kernel, strides=self.strides,
                    padding=self.padding, **self.kwargs))
            elif i != 0 and i % 2 == 0:
                setattr(self, f"conv_{i}", tensorflow.keras.layers.Conv2D(
                    filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same', **self.kwargs))
            else:
                setattr(self, f"conv_{i}", tensorflow.keras.layers.Conv2D(
                    filters=2 * self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same', **self.kwargs))
            if self.include_bn_activation:
                setattr(self, f"bn_{i}", tensorflow.keras.layers.BatchNormalization(
                    momentum=0.03 if self.mode == "YOLOv5" else 0.99))
                if activation == 'LeakyReLU':
                    setattr(self, f"act_{i}", tensorflow.keras.layers.LeakyReLU(alpha=0.1))
                if activation == 'Mish':
                    setattr(self, f"act_{i}", Mish())
                if activation == 'Swish':
                    setattr(self, f"act_{i}", tensorflow.keras.layers.Activation('swish'))

    def call(self, x, training=True, **kwargs):
        for i in range(self.num_conv):
            x = getattr(self, f"conv_{i}")(x)
            if self.include_bn_activation:
                x = getattr(self, f"bn_{i}")(x)
                x = getattr(self, f"act_{i}")(x)
        return x

    def get_config(self):
        config = {
            'mode': self.mode,
            'filters': self.filters,
            'num_conv': self.num_conv,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'first_conv_strides': self.strides,
            'first_conv_kernel': self.kernel,
            'first_conv_padding': self.padding,
            'include_bn_activation': self.include_bn_activation
        }
        base_config = super(YOLOConvBlock, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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


class BatchNormalization(BatchNormalization):

    def __init__(self, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=False, **kwargs):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(inputs, training)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        return config


class DarkNetConvolutional(Layer):

    def __init__(self, filters=32, kernel_size=(3, 3), downsample=False, activate=True, bn=True,
                 activate_type='LeakyReLU', **kwargs):
        super(DarkNetConvolutional, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.activate = activate
        self.activate_type = activate_type
        self.bn = bn
        if self.downsample:
            self.zero = tensorflow.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
            self.padding = 'valid'
            self.strides = (2, 2)
        else:
            self.strides = (1, 1)
            self.padding = 'same'

        self.convolutional = tensorflow.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=not self.bn,
            kernel_regularizer=tensorflow.keras.regularizers.l2(0.0005),
            kernel_initializer=tensorflow.random_normal_initializer(stddev=0.01),
            bias_initializer=tensorflow.constant_initializer(0.)
        )
        if self.bn:
            self.bn_conv = BatchNormalization()
        if self.activate:
            if self.activate_type == "LeakyReLU":
                self.activation = tensorflow.keras.layers.LeakyReLU(alpha=0.1)
            elif self.activate_type == "Mish":
                self.activation = Mish()

    def call(self, inputs, **kwargs):
        if self.downsample:
            inputs = self.zero(inputs)
        conv = self.convolutional(inputs)
        if self.bn:
            conv = self.bn_conv(conv)
        if self.activate:
            conv = self.activation(conv)
        return conv

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'downsample': self.downsample,
            'activate': self.activate,
            'bn': self.bn,
            'activate_type': self.activate_type,
        }
        base_config = super(DarkNetConvolutional, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DarkNetResBlock(Layer):

    def __init__(self, filter_num1=32, filter_num2=32, activate_type='LeakyReLU', **kwargs):
        super(DarkNetResBlock, self).__init__(**kwargs)
        self.filter_num1 = filter_num1
        self.filter_num2 = filter_num2
        self.activate_type = activate_type

        self.conv_1 = DarkNetConvolutional(
            filters=filter_num1,
            kernel_size=(1, 1),
            activate_type=self.activate_type
        )
        self.conv_2 = DarkNetConvolutional(
            filters=filter_num2,
            kernel_size=(3, 3),
            activate_type=self.activate_type
        )

    def call(self, inputs, **kwargs):
        short_cut = inputs
        conv = self.conv_1(inputs)
        conv = self.conv_2(conv)
        residual_output = short_cut + conv
        return residual_output

    def get_config(self):
        config = {
            'filter_num1': self.filter_num1,
            'filter_num2': self.filter_num2,
            'activate_type': self.activate_type,
        }
        base_config = super(DarkNetResBlock, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DarkNetUpsample(Layer):

    def __init__(self, **kwargs):
        super(DarkNetUpsample, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')

    def get_config(self):
        config = super(DarkNetUpsample, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CONVBlock(Layer):
    """Conv block layer """

    def __init__(
            self, n_conv_layers=2, filters=16, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='RandomNormal',
            dilation=(1, 1), padding='same', activation='relu', transpose=False, use_bias=True,
            use_activation_layer=False, leaky_relu_layer=True, leaky_relu_alpha=0.3,
            normalization='batch', dropout_layer=True, dropout_rate=0.1, kernel_regularizer=None,
            layers_seq_config: str = 'conv_bn_lrelu_drop_conv_bn_lrelu_drop',
            **kwargs
    ):

        super(CONVBlock, self).__init__(**kwargs)
        self.n_conv_layers = n_conv_layers
        self.filters = filters
        self.activation = None if leaky_relu_layer else activation
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.padding = padding
        self.normalization = normalization
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate
        self.leaky_relu_layer = leaky_relu_layer
        self.layers_seq_config = layers_seq_config
        self.leaky_relu_alpha = leaky_relu_alpha
        self.use_activation_layer = use_activation_layer
        self.kernel_initializer = kernel_initializer
        self.transpose = transpose
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer

        conv_activation = None if self.use_activation_layer else self.activation
        for i in range(self.n_conv_layers):
            strides = self.strides if i + 1 == self.n_conv_layers else (1, 1)
            if self.transpose:
                setattr(
                    self, f"conv_{i}",
                    layers.Conv2DTranspose(
                        filters=self.filters, kernel_size=self.kernel_size, strides=strides,
                        padding=self.padding, activation=conv_activation, dilation_rate=self.dilation,
                        use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer
                    )
                )
            else:
                setattr(
                    self, f"conv_{i}",
                    layers.Conv2D(
                        filters=self.filters, kernel_size=self.kernel_size, strides=strides,
                        padding=self.padding, activation=conv_activation, dilation_rate=self.dilation,
                        use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer
                    )
                )
            if self.normalization == "batch":
                setattr(self, f"norm_{i}", layers.BatchNormalization(axis=-1))
            if self.normalization == "instance":
                setattr(self, f"norm_{i}", InstanceNormalization(axis=-1))
            if self.use_activation_layer and self.activation:
                setattr(self, f'activ_{i}', layers.Activation(self.activation))
            if self.leaky_relu_layer:
                setattr(self, f'leaky_relu_{i}', layers.LeakyReLU(alpha=self.leaky_relu_alpha))
            if self.dropout_layer:
                setattr(self, f'drop_{i}', layers.Dropout(rate=self.dropout_rate))

    def call(self, input_, training=True, **kwargs):

        if not isinstance(input_, (np.int32, np.float64, np.float32, np.float16)):
            input_ = cast(input_, 'float16')
        x = input_
        if self.layers_seq_config == 'conv_conv_bn_LRelu_drop':
            for i in range(self.n_conv_layers):
                x = getattr(self, f'conv_{i}')(x)
            if self.normalization:
                x = getattr(self, f'norm_0')(x)
            if self.use_activation_layer:
                x = getattr(self, f'activ_0')(x)
            if self.leaky_relu_layer:
                x = getattr(self, f'leaky_relu_0')(x)
            if self.dropout_layer:
                x = getattr(self, f'drop_0')(x)
        else:
            for i in range(self.n_conv_layers):
                x = getattr(self, f'conv_{i}')(x)
                # print(f"{getattr(self, f'conv_{i}').name}: {x.shape}, {getattr(self, f'conv_{i}').strides}")
                if self.normalization:
                    x = getattr(self, f'norm_{i}')(x)
                    # print(f"norm_{i}: {x.shape}")
                if self.use_activation_layer:
                    x = getattr(self, f'activ_{i}')(x)
                    # print(f"activ_{i}: {x.shape}")
                if self.leaky_relu_layer:
                    x = getattr(self, f'leaky_relu_{i}')(x)
                    # print(f"leaky_relu_{i}: {x.shape}")
                if self.dropout_layer:
                    x = getattr(self, f'drop_{i}')(x)
        return x

    def get_config(self):
        config = {
            'n_conv_layers': self.n_conv_layers,
            'filters': self.filters,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation': self.dilation,
            'padding': self.padding,
            'normalization': self.normalization,
            'dropout_layer': self.dropout_layer,
            'dropout_rate': self.dropout_rate,
            'leaky_relu_layer': self.leaky_relu_layer,
            'leaky_relu_alpha': self.leaky_relu_alpha,
            'layers_seq_config': self.layers_seq_config,
            'use_activation_layer': self.use_activation_layer,
            'kernel_initializer': self.kernel_initializer,
            'transpose': self.transpose,
            'use_bias': self.use_bias,
            'kernel_regularizer': self.kernel_regularizer
        }
        base_config = super(CONVBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        if self.transpose:
            if self.padding == 'same':
                shape_1 = input_shape[1] * self.strides[0]
                shape_2 = input_shape[2] * self.strides[1]
            else:
                add_1 = self.kernel_size[0] - self.strides[0] if self.kernel_size[0] > self.strides[0] else 0
                shape_1 = input_shape[1] * self.strides[0] + add_1
                add_2 = self.kernel_size[1] - self.strides[1] if self.kernel_size[1] > self.strides[1] else 0
                shape_2 = input_shape[2] * self.strides[1] + add_2
            output_shape = (None, shape_1, shape_2, self.filters)
        else:
            if self.padding == 'same':
                shape_1 = input_shape[1] // self.strides[0] + 1 if input_shape[1] % self.strides[0] > 0 \
                    else input_shape[1] // self.strides[0]
                shape_2 = input_shape[2] // self.strides[1] + 1 if input_shape[2] % self.strides[1] > 0 \
                    else input_shape[2] // self.strides[1]
            else:
                shape_1 = (input_shape[1] - self.kernel_size[0]) // self.strides[0] + 1 \
                    if (input_shape[1] - self.kernel_size[0]) % self.strides[0] > 0 \
                    else (input_shape[1] - self.kernel_size[0]) // self.strides[0]
                shape_2 = (input_shape[2] - self.kernel_size[1]) // self.strides[1] + 1 \
                    if (input_shape[2] - self.kernel_size[1]) % self.strides[1] > 0 \
                    else (input_shape[2] - self.kernel_size[1]) // self.strides[1]
            output_shape = (None, shape_1, shape_2, self.filters)
        return output_shape


class PSPBlock2D(Layer):
    """
    PSP Block2D layer
    n_pooling_branches - defines amt of pooling/upsampling operations
    filters_coef - defines the multiplication factor for amt of filters in pooling branches
    n_conv_layers - number of conv layers in one downsampling/upsampling segment
    """

    def __init__(self, filters_base=16, n_pooling_branches=2, filters_coef=1, n_conv_layers=2, activation='relu',
                 kernel_size=(3, 3), batch_norm_layer=True, dropout_layer=True,
                 dropout_rate=0.1, **kwargs):

        super(PSPBlock2D, self).__init__(**kwargs)
        self.filters = filters_base
        self.n_pooling_branches = n_pooling_branches
        self.filters_coef = filters_coef
        self.n_conv_layers = n_conv_layers
        self.activation = activation
        self.kernel_size = kernel_size
        self.batch_norm_layer = batch_norm_layer
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        self.conv_start = layers.Conv2D(filters=self.filters_coef * self.filters, kernel_size=self.kernel_size,
                                        padding='same', activation=self.activation, data_format='channels_last',
                                        groups=1, use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

        for i in range(0, self.n_pooling_branches):
            setattr(self, f"maxpool_{i}",
                    layers.MaxPool2D(pool_size=2 ** i, padding='same'))
            for j in range(self.n_conv_layers):
                setattr(self, f"conv_{i, j}",
                        layers.Conv2D(filters=self.filters_coef * self.filters * (i + 1), kernel_size=self.kernel_size,
                                      padding='same', activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            setattr(self, f"convtranspose_{i}",
                    layers.Conv2DTranspose(filters=self.filters_coef * self.filters * (i + 1),
                                           kernel_size=(1, 1), strides=2 ** i, padding='same',
                                           activation=self.activation, data_format='channels_last'))
            if self.batch_norm_layer:
                setattr(self, f"batchnorm_{i}", layers.BatchNormalization())
            if self.dropout_layer:
                setattr(self, f"dropout_{i}", layers.Dropout(rate=self.dropout_rate))

        self.concatenate = layers.Concatenate()

        self.conv_end = layers.Conv2D(filters=self.filters_coef * self.filters, kernel_size=self.kernel_size,
                                      padding='same', activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    def call(self, input_, training=True, **kwargs):

        if not isinstance(input_, (np.int32, np.float64, np.float32, np.float16)):
            input_ = cast(input_, 'float16')

        x = self.conv_start(input_)
        conc_list = []
        for i in range(0, self.n_pooling_branches):
            setattr(self, f'x_{i}', getattr(self, f'maxpool_{i}')(x))
            for j in range(self.n_conv_layers):
                setattr(self, f'x_{i}', getattr(self, f'conv_{i, j}')(getattr(self, f'x_{i}')))
            if self.batch_norm_layer:
                setattr(self, f'x_{i}', getattr(self, f'batchnorm_{i}')(getattr(self, f'x_{i}')))
            if self.dropout_layer:
                setattr(self, f'x_{i}', getattr(self, f'dropout_{i}')(getattr(self, f'x_{i}')))
            setattr(self, f'x_{i}', getattr(self, f'convtranspose_{i}')(getattr(self, f'x_{i}')))
            # setattr(self, f'x_{i}', layers.CenterCrop(input_.shape[1], input_.shape[2])(getattr(self, f'x_{i}')))
            conc_list.append(getattr(self, f'x_{i}'))

        concat = self.concatenate(conc_list)
        x = self.conv_end(concat)
        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'n_pooling_branches': self.n_pooling_branches,
            'filters_coef': self.filters_coef,
            'n_conv_layers': self.n_conv_layers,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation': self.dilation,
            'padding': self.padding,
            'batch_norm_layer': self.batch_norm_layer,
            'dropout_layer': self.dropout_layer,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(PSPBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UNETBlock2D(Layer):
    """
    UNET Block2D layer
    n_pooling_branches - defines amt of downsampling/upsampling operations
    filters_coef - defines the multiplication factor for amt of filters in pooling branches
    n_conv_layers - number of conv layers in one downsampling/upsampling segment
    """

    def __init__(
            self, filters_base=16, n_pooling_branches=2, filters_coef=1,
            n_conv_layers=2, kernel_size=(3, 3), kernel_initializer='RandomNormal',
            activation='relu', use_bias=True, maxpooling=True, upsampling=True,
            use_activation_layer=False, leaky_relu_layer=True, leaky_relu_alpha=0.3,
            normalization='batch', dropout_layer=True, dropout_rate=0.1,
            kernel_regularizer=None, **kwargs
    ):

        super(UNETBlock2D, self).__init__(**kwargs)
        self.filters = filters_base
        self.n_pooling_branches = n_pooling_branches
        self.filters_coef = filters_coef
        self.n_conv_layers = n_conv_layers
        self.activation = activation
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.use_activation_layer = use_activation_layer
        self.leaky_relu_layer = leaky_relu_layer
        self.leaky_relu_alpha = leaky_relu_alpha
        self.normalization = normalization
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate
        self.maxpooling = maxpooling
        self.upsampling = upsampling
        self.kernel_regularizer = kernel_regularizer

        down_strides = (1, 1) if self.maxpooling else (2, 2)
        up_strides = (1, 1) if self.upsampling else (2, 2)
        transpose = False if self.upsampling else True
        setattr(
            self, f"start_block",
            CONVBlock(
                n_conv_layers=self.n_conv_layers, filters=self.filters * self.filters_coef,
                kernel_size=self.kernel_size, strides=(1, 1), kernel_initializer=self.kernel_initializer,
                activation=self.activation, transpose=False, use_bias=self.use_bias,
                use_activation_layer=self.use_activation_layer, leaky_relu_layer=self.leaky_relu_layer,
                leaky_relu_alpha=self.leaky_relu_alpha, normalization=self.normalization,
                dropout_layer=self.dropout_layer, dropout_rate=self.dropout_rate,
                layers_seq_config='conv_bn_lrelu_drop_conv_bn_lrelu_drop',
                kernel_regularizer=self.kernel_regularizer
            )
        )
        for i in range(self.n_pooling_branches):
            down_filters = 1024 if self.filters * (2 ** (i + 1)) * self.filters_coef > 1024 else \
                self.filters * (2 ** (i + 1)) * self.filters_coef
            setattr(
                self, f"downblock_{i + 1}",
                CONVBlock(
                    n_conv_layers=self.n_conv_layers, filters=down_filters, kernel_size=self.kernel_size,
                    strides=down_strides, kernel_initializer=self.kernel_initializer,
                    activation=self.activation, transpose=False, use_bias=self.use_bias,
                    use_activation_layer=self.use_activation_layer, leaky_relu_layer=self.leaky_relu_layer,
                    leaky_relu_alpha=self.leaky_relu_alpha, normalization=self.normalization,
                    dropout_layer=self.dropout_layer, dropout_rate=self.dropout_rate,
                    layers_seq_config='conv_bn_lrelu_drop_conv_bn_lrelu_drop',
                    kernel_regularizer=self.kernel_regularizer
                )
            )
            if self.maxpooling:
                setattr(self, f"maxpool_{i + 1}", layers.MaxPool2D(pool_size=2, padding='same'))
        for i in range(self.n_pooling_branches):
            up_filters = 1024 if self.filters * (2 ** (self.n_pooling_branches - 1 - i)) * self.filters_coef > 1024 \
                else self.filters * (2 ** (self.n_pooling_branches - 1 - i)) * self.filters_coef
            setattr(
                self, f"upblock_{i + 1}",
                CONVBlock(
                    n_conv_layers=self.n_conv_layers, filters=up_filters, kernel_size=self.kernel_size,
                    strides=up_strides, kernel_initializer=self.kernel_initializer,
                    activation=self.activation, transpose=transpose, use_bias=self.use_bias,
                    use_activation_layer=self.use_activation_layer, leaky_relu_layer=self.leaky_relu_layer,
                    leaky_relu_alpha=self.leaky_relu_alpha, normalization=self.normalization,
                    dropout_layer=self.dropout_layer, dropout_rate=self.dropout_rate,
                    layers_seq_config='conv_bn_lrelu_drop_conv_bn_lrelu_drop',
                    kernel_regularizer=self.kernel_regularizer
                )
            )
            if self.upsampling:
                setattr(self, f"upsample_{i + 1}", layers.UpSampling2D(size=2))

    def call(self, input_, training=True, **kwargs):
        if not isinstance(input_, (np.int32, np.float64, np.float32, np.float16)):
            input_ = cast(input_, 'float16')

        setattr(self, f'start', getattr(self, f'start_block')(input_))
        for i in range(self.n_pooling_branches):
            if i == 0:
                setattr(self, f'down_{i + 1}', getattr(self, f'downblock_{i + 1}')(getattr(self, f'start')))
            else:
                uplink = getattr(self, f'mp_{i}') if self.maxpooling else getattr(self, f'down_{i}')
                setattr(self, f'down_{i + 1}', getattr(self, f'downblock_{i + 1}')(uplink))
            if self.maxpooling:
                setattr(self, f'mp_{i + 1}', getattr(self, f'maxpool_{i + 1}')(getattr(self, f'down_{i + 1}')))

        for i in range(self.n_pooling_branches):
            if i == 0:
                uplink = getattr(self, f'mp_{self.n_pooling_branches}') if self.maxpooling \
                    else getattr(self, f'down_{self.n_pooling_branches}')
                setattr(self, f'up_{i + 1}', getattr(self, f'upblock_{i + 1}')(uplink))
            else:
                setattr(self, f'up_{i + 1}', getattr(self, f'upblock_{i + 1}')(getattr(self, f'concat_{i}')))
            if self.upsampling:
                setattr(self, f'us_{i + 1}', getattr(self, f'upsample_{i + 1}')(getattr(self, f'up_{i + 1}')))
            concatlink_1 = getattr(self, f'us_{i + 1}') if self.upsampling else getattr(self, f'up_{i + 1}')
            if i != self.n_pooling_branches - 1:
                concatlink_2 = getattr(self, f'mp_{self.n_pooling_branches - 1 - i}') if self.maxpooling \
                    else getattr(self, f'down_{self.n_pooling_branches - 1 - i}')
            else:
                concatlink_2 = getattr(self, f'start')
            setattr(self, f'concat_{i + 1}', layers.Concatenate()([concatlink_1, concatlink_2]))

        return getattr(self, f'concat_{self.n_pooling_branches}')

    def get_config(self):
        config = {
            'filters_base': self.filters,
            'n_pooling_branches': self.n_pooling_branches,
            'filters_coef': self.filters_coef,
            'n_conv_layers': self.n_conv_layers,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'dropout_layer': self.dropout_layer,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
            'use_bias': self.use_bias,
            'use_activation_layer': self.use_activation_layer,
            'leaky_relu_layer': self.leaky_relu_layer,
            'leaky_relu_alpha': self.leaky_relu_alpha,
            'normalization': self.normalization,
            'maxpooling': self.maxpooling,
            'upsampling': self.upsampling,
            'kernel_regularizer': self.kernel_regularizer
        }
        base_config = super(UNETBlock2D, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        output_shape = (None, input_shape[1], input_shape[2], self.filters * 2)
        return output_shape


class UNETBlock1D(Layer):
    """
    UNET Block1D layer
    n_pooling_branches - defines amt of downsampling/upsampling operations
    filters_coef - defines the multiplication factor for amt of filters in pooling branches
    n_conv_layers - number of conv layers in one downsampling/upsampling segment
    """

    def __init__(self, filters_base=16, n_pooling_branches=2, filters_coef=1, n_conv_layers=2, activation='relu',
                 kernel_size=5, batch_norm_layer=True,
                 dropout_layer=True, dropout_rate=0.1, **kwargs):

        super(UNETBlock1D, self).__init__(**kwargs)
        self.filters = filters_base
        self.n_pooling_branches = n_pooling_branches
        self.filters_coef = filters_coef
        self.n_conv_layers = n_conv_layers
        self.activation = activation
        self.kernel_size = kernel_size
        self.batch_norm_layer = batch_norm_layer
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        setattr(self, f"start_conv",
                layers.Conv1D(filters=self.filters * self.filters_coef, kernel_size=self.kernel_size,
                              activation=self.activation, data_format='channels_last',
                              groups=1, use_bias=True, padding='same',
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                              activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

        for i in range(0, self.n_pooling_branches):
            if i == 0:
                for j in range(1, self.n_conv_layers):
                    setattr(self, f"conv_d{i}_{j}",
                            layers.Conv1D(filters=self.filters * (i + 1) * self.filters_coef,
                                          kernel_size=self.kernel_size, padding='same',
                                          activation=self.activation, data_format='channels_last',
                                          groups=1, use_bias=True,
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                          activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            else:
                for j in range(0, self.n_conv_layers):
                    setattr(self, f"conv_d{i}_{j}",
                            layers.Conv1D(filters=self.filters * (i + 1) * self.filters_coef,
                                          kernel_size=self.kernel_size, padding='same',
                                          activation=self.activation, data_format='channels_last',
                                          groups=1, use_bias=True,
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                          activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            if self.batch_norm_layer:
                setattr(self, f"batchnorm_d{i}", layers.BatchNormalization())
            if self.dropout_layer:
                setattr(self, f'dropout_{i}', layers.Dropout(rate=self.dropout_rate))

            setattr(self, f"maxpool_{i}",
                    layers.MaxPool1D(pool_size=2, padding='same'))

        for i in range(self.n_pooling_branches, self.n_pooling_branches * 2):
            setattr(self, f"upsample_{i}",
                    layers.UpSampling1D(size=2))
            for j in range(0, self.n_conv_layers):
                setattr(self, f"conv_u{i}_{j}",
                        layers.Conv1D(filters=self.filters * (2 * self.n_pooling_branches - i) * self.filters_coef,
                                      kernel_size=self.kernel_size, padding='same',
                                      activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            if self.batch_norm_layer:
                setattr(self, f"batchnorm_u{i}", layers.BatchNormalization())
            setattr(self, f"concatenate_{i}", layers.Concatenate())

        for i in range(self.n_conv_layers):
            setattr(self, f"conv_bottom{i}",
                    layers.Conv1D(filters=2 * self.filters * self.n_pooling_branches * self.filters_coef,
                                  kernel_size=self.kernel_size, padding='same',
                                  activation=self.activation, data_format='channels_last',
                                  groups=1, use_bias=True,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    def call(self, input_, training=True, **kwargs):
        if not isinstance(input_, (np.int32, np.float64, np.float32, np.float16)):
            input_ = cast(input_, 'float16')
        concList = [[] for i in range(self.n_pooling_branches)]

        for i in range(0, self.n_pooling_branches):  # Подумать над конкатенайт и кроп
            if i == 0:
                setattr(self, f'x_{i}', getattr(self, f'start_conv')(input_))

                for j in range(1, self.n_conv_layers):
                    setattr(self, f'x_{i}', getattr(self, f'conv_d{i}_{j}')(getattr(self, f'x_{i}')))

            else:
                for j in range(0, self.n_conv_layers):
                    setattr(self, f'x_{i}', getattr(self, f'conv_d{i}_{j}')(getattr(self, f'x_{i}')))

            if self.batch_norm_layer:
                setattr(self, f'x_{i}', getattr(self, f'batchnorm_d{i}')(getattr(self, f'x_{i}')))

            if self.dropout_layer:
                setattr(self, f'x_{i}', getattr(self, f'dropout_{i}')(getattr(self, f'x_{i}')))

            concList[i].append(getattr(self, f'x_{i}'))

            setattr(self, f'x_{i + 1}', getattr(self, f'maxpool_{i}')(getattr(self, f'x_{i}')))

        for i in range(0, self.n_conv_layers):
            setattr(self, f'x_{self.n_pooling_branches}',
                    getattr(self, f'conv_bottom{i}')(getattr(self, f'x_{self.n_pooling_branches}')))

        for i in range(self.n_pooling_branches, self.n_pooling_branches * 2):

            setattr(self, f'x_{i}', getattr(self, f"upsample_{i}")(getattr(self, f'x_{i}')))
            concList[2 * self.n_pooling_branches - i - 1].append(getattr(self, f'x_{i}'))
            setattr(self, f'x_{i}', getattr(self, f"concatenate_{i}")(concList[2 * self.n_pooling_branches - i - 1]))

            if self.batch_norm_layer:
                for j in range(0, self.n_conv_layers):
                    setattr(self, f'x_{i}', getattr(self, f'conv_u{i}_{j}')(getattr(self, f'x_{i}')))
                setattr(self, f'x_{i + 1}', getattr(self, f'batchnorm_u{i}')(getattr(self, f'x_{i}')))
            else:
                for j in range(0, self.n_conv_layers):
                    if j != self.n_conv_layers - 1:
                        setattr(self, f'x_{i}', getattr(self, f'conv_u{i}_{j}')(getattr(self, f'x_{i}')))
                    else:
                        setattr(self, f'x_{i + 1}', getattr(self, f'conv_u{i}_{j}')(getattr(self, f'x_{i}')))

        x = getattr(self, f'x_{i + 1}')
        return x

    def get_config(self):
        config = {
            'filters_base': self.filters,
            'n_pooling_branches': self.n_pooling_branches,
            'filters_coef': self.filters_coef,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation': self.dilation,
            'padding': self.padding,
            'batch_norm_layer': self.batch_norm_layer,
            'dropout_layer': self.dropout_layer,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(UNETBlock1D, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UNETBlock3D(Layer):
    """
    UNET Block3D layer
    n_pooling_branches - defines amt of downsampling/upsampling operations
    filters_coef - defines the multiplication factor for amt of filters in pooling branches
    n_conv_layers - number of conv layers in one downsampling/upsampling segment
    """

    def __init__(self, filters_base=16, n_pooling_branches=2, filters_coef=1, n_conv_layers=2, activation='relu',
                 kernel_size=(3, 3, 3), batch_norm_layer=True,
                 dropout_layer=True, dropout_rate=0.1, **kwargs):

        super(UNETBlock3D, self).__init__(**kwargs)
        self.filters = filters_base
        self.n_pooling_branches = n_pooling_branches
        self.filters_coef = filters_coef
        self.n_conv_layers = n_conv_layers
        self.activation = activation
        self.kernel_size = kernel_size
        self.batch_norm_layer = batch_norm_layer
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        setattr(self, f"start_conv",
                layers.Conv3D(filters=self.filters * self.filters_coef, kernel_size=self.kernel_size,
                              activation=self.activation, data_format='channels_last',
                              groups=1, use_bias=True, padding='same',
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                              activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

        for i in range(0, self.n_pooling_branches):
            if i == 0:
                for j in range(1, self.n_conv_layers):
                    setattr(self, f"conv_d{i}_{j}",
                            layers.Conv3D(filters=self.filters * (i + 1) * self.filters_coef,
                                          kernel_size=self.kernel_size, padding='same',
                                          activation=self.activation, data_format='channels_last',
                                          groups=1, use_bias=True,
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                          activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            else:
                for j in range(0, self.n_conv_layers):
                    setattr(self, f"conv_d{i}_{j}",
                            layers.Conv3D(filters=self.filters * (i + 1) * self.filters_coef,
                                          kernel_size=self.kernel_size, padding='same',
                                          activation=self.activation, data_format='channels_last',
                                          groups=1, use_bias=True,
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                          activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            if self.batch_norm_layer:
                setattr(self, f"batchnorm_d{i}", layers.BatchNormalization())
            if self.dropout_layer:
                setattr(self, f'dropout_{i}', layers.Dropout(rate=self.dropout_rate))

            setattr(self, f"maxpool_{i}",
                    layers.MaxPool3D(pool_size=2, padding='same'))

        for i in range(self.n_pooling_branches, self.n_pooling_branches * 2):
            setattr(self, f"upsample_{i}",
                    layers.UpSampling3D(size=2))
            for j in range(0, self.n_conv_layers):
                setattr(self, f"conv_u{i}_{j}",
                        layers.Conv3D(filters=self.filters * (2 * self.n_pooling_branches - i) * self.filters_coef,
                                      kernel_size=self.kernel_size, padding='same',
                                      activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            if self.batch_norm_layer:
                setattr(self, f"batchnorm_u{i}", layers.BatchNormalization())
            setattr(self, f"concatenate_{i}", layers.Concatenate())

        for i in range(self.n_conv_layers):
            setattr(self, f"conv_bottom{i}",
                    layers.Conv3D(filters=2 * self.filters * self.n_pooling_branches * self.filters_coef,
                                  kernel_size=self.kernel_size, padding='same',
                                  activation=self.activation, data_format='channels_last',
                                  groups=1, use_bias=True,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    def call(self, input_, training=True, **kwargs):
        if not isinstance(input_, (np.int32, np.float64, np.float32, np.float16)):
            input_ = cast(input_, 'float16')
        concList = [[] for i in range(self.n_pooling_branches)]

        for i in range(0, self.n_pooling_branches):  # Подумать над конкатенайт и кроп
            if i == 0:
                setattr(self, f'x_{i}', getattr(self, f'start_conv')(input_))

                for j in range(1, self.n_conv_layers):
                    setattr(self, f'x_{i}', getattr(self, f'conv_d{i}_{j}')(getattr(self, f'x_{i}')))

            else:
                for j in range(0, self.n_conv_layers):
                    setattr(self, f'x_{i}', getattr(self, f'conv_d{i}_{j}')(getattr(self, f'x_{i}')))

            if self.batch_norm_layer:
                setattr(self, f'x_{i}', getattr(self, f'batchnorm_d{i}')(getattr(self, f'x_{i}')))

            if self.dropout_layer:
                setattr(self, f'x_{i}', getattr(self, f'dropout_{i}')(getattr(self, f'x_{i}')))

            concList[i].append(getattr(self, f'x_{i}'))

            setattr(self, f'x_{i + 1}', getattr(self, f'maxpool_{i}')(getattr(self, f'x_{i}')))

        for i in range(0, self.n_conv_layers):
            setattr(self, f'x_{self.n_pooling_branches}',
                    getattr(self, f'conv_bottom{i}')(getattr(self, f'x_{self.n_pooling_branches}')))

        for i in range(self.n_pooling_branches, self.n_pooling_branches * 2):

            setattr(self, f'x_{i}', getattr(self, f"upsample_{i}")(getattr(self, f'x_{i}')))
            concList[2 * self.n_pooling_branches - i - 1].append(getattr(self, f'x_{i}'))
            setattr(self, f'x_{i}', getattr(self, f"concatenate_{i}")(concList[2 * self.n_pooling_branches - i - 1]))

            if self.batch_norm_layer:
                for j in range(0, self.n_conv_layers):
                    setattr(self, f'x_{i}', getattr(self, f'conv_u{i}_{j}')(getattr(self, f'x_{i}')))
                setattr(self, f'x_{i + 1}', getattr(self, f'batchnorm_u{i}')(getattr(self, f'x_{i}')))
            else:
                for j in range(0, self.n_conv_layers):
                    if j != self.n_conv_layers - 1:
                        setattr(self, f'x_{i}', getattr(self, f'conv_u{i}_{j}')(getattr(self, f'x_{i}')))
                    else:
                        setattr(self, f'x_{i + 1}', getattr(self, f'conv_u{i}_{j}')(getattr(self, f'x_{i}')))

        x = getattr(self, f'x_{i + 1}')
        return x

    def get_config(self):
        config = {
            'filters_base': self.filters_base,
            'n_pooling_branches': self.n_pooling_branches,
            'filters_coef': self.filters_coef,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'batch_norm_layer': self.batch_norm_layer,
            'dropout_layer': self.dropout_layer,
            'n_conv_layers': self.n_conv_layers
        }
        base_config = super(UNETBlock3D, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PSPBlock1D(Layer):
    """
    PSP Block1D layer
    n_pooling_branches - defines amt of pooling/upsampling operations
    filters_coef - defines the multiplication factor for amt of filters in pooling branches
    n_conv_layers - number of conv layers in one downsampling/upsampling segment
    """

    def __init__(self, filters_base=16, n_pooling_branches=2, filters_coef=1, n_conv_layers=2, activation='relu',
                 kernel_size=(3, 3), batch_norm_layer=True, dropout_layer=True,
                 dropout_rate=0.1, **kwargs):

        super(PSPBlock1D, self).__init__(**kwargs)
        self.filters = filters_base
        self.n_pooling_branches = n_pooling_branches
        self.filters_coef = filters_coef
        self.n_conv_layers = n_conv_layers
        self.activation = activation
        self.kernel_size = kernel_size
        self.batch_norm_layer = batch_norm_layer
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        self.conv_start = layers.Conv1D(filters=self.filters_coef * self.filters, kernel_size=self.kernel_size,
                                        padding='same', activation=self.activation, data_format='channels_last',
                                        groups=1, use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

        for i in range(0, self.n_pooling_branches):
            setattr(self, f"maxpool_{i}",
                    layers.MaxPool1D(pool_size=2 ** i, padding='same'))
            for j in range(self.n_conv_layers):
                setattr(self, f"conv_{i, j}",
                        layers.Conv1D(filters=self.filters_coef * self.filters * (i + 1), kernel_size=self.kernel_size,
                                      padding='same', activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            setattr(self, f"convtranspose_{i}",
                    layers.Conv1DTranspose(filters=self.filters_coef * self.filters * (i + 1),
                                           kernel_size=1, strides=2 ** i, padding='same',
                                           activation=self.activation, data_format='channels_last'))
            if self.batch_norm_layer:
                setattr(self, f"batchnorm_{i}", layers.BatchNormalization())
            if self.dropout_layer:
                setattr(self, f"dropout_{i}", layers.Dropout(rate=self.dropout_rate))

        self.concatenate = layers.Concatenate()

        self.conv_end = layers.Conv1D(filters=self.filters_coef * self.filters, kernel_size=self.kernel_size,
                                      padding='same', activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    def call(self, input_, training=True, **kwargs):

        if not isinstance(input_, (np.int32, np.float64, np.float32, np.float16)):
            input_ = cast(input_, 'float16')

        x = self.conv_start(input_)

        conc_list = []

        for i in range(0, self.n_pooling_branches):
            setattr(self, f'x_{i}', getattr(self, f'maxpool_{i}')(x))
            for j in range(self.n_conv_layers):
                setattr(self, f'x_{i}', getattr(self, f'conv_{i, j}')(getattr(self, f'x_{i}')))
            if self.batch_norm_layer:
                setattr(self, f'x_{i}', getattr(self, f'batchnorm_{i}')(getattr(self, f'x_{i}')))
            if self.dropout_layer:
                setattr(self, f'x_{i}', getattr(self, f'dropout_{i}')(getattr(self, f'x_{i}')))
            setattr(self, f'x_{i}', getattr(self, f'convtranspose_{i}')(getattr(self, f'x_{i}')))
            # setattr(self, f'x_{i}', layers.CenterCrop(input_.shape[1], input_.shape[2])(getattr(self, f'x_{i}')))
            conc_list.append(getattr(self, f'x_{i}'))

        concat = self.concatenate(conc_list)
        x = self.conv_end(concat)
        return x

    def get_config(self):
        config = {
            'filters_base': self.filters_base,
            'n_pooling_branches': self.n_pooling_branches,
            'filters_coef': self.filters_coef,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'batch_norm_layer': self.batch_norm_layer,
            'dropout_layer': self.dropout_layer,
            'n_conv_layers': self.n_conv_layers
        }
        base_config = super(PSPBlock1D, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PSPBlock3D(Layer):
    """
    PSP Block3D layer
    n_pooling_branches - defines amt of pooling/upsampling operations
    filters_coef - defines the multiplication factor for amt of filters in pooling branches
    n_conv_layers - number of conv layers in one downsampling/upsampling segment
    """

    def __init__(self, filters_base=16, n_pooling_branches=2, filters_coef=1, n_conv_layers=1, activation='relu',
                 kernel_size=(3, 3, 3), batch_norm_layer=True, dropout_layer=True,
                 dropout_rate=0.1, **kwargs):

        super(PSPBlock3D, self).__init__(**kwargs)
        self.filters = filters_base
        self.n_pooling_branches = n_pooling_branches
        self.filters_coef = filters_coef
        self.n_conv_layers = n_conv_layers
        self.activation = activation
        self.kernel_size = kernel_size
        self.batch_norm_layer = batch_norm_layer
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        self.conv_start = layers.Conv3D(filters=self.filters_coef * self.filters, kernel_size=self.kernel_size,
                                        padding='same', activation=self.activation, data_format='channels_last',
                                        groups=1, use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

        for i in range(0, self.n_pooling_branches):
            setattr(self, f"maxpool_{i}",
                    layers.MaxPool3D(pool_size=2 ** i, padding='same'))
            for j in range(self.n_conv_layers):
                setattr(self, f"conv_{i, j}",
                        layers.Conv3D(filters=self.filters_coef * self.filters * (i + 1), kernel_size=self.kernel_size,
                                      padding='same', activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            setattr(self, f"convtranspose_{i}",
                    layers.Conv3DTranspose(filters=self.filters_coef * self.filters * (i + 1),
                                           kernel_size=(1, 1, 1), strides=2 ** i, padding='same',
                                           activation=self.activation, data_format='channels_last'))
            if self.batch_norm_layer:
                setattr(self, f"batchnorm_{i}", layers.BatchNormalization())
            if self.dropout_layer:
                setattr(self, f"dropout_{i}", layers.Dropout(rate=self.dropout_rate))

        self.concatenate = layers.Concatenate()

        self.conv_end = layers.Conv3D(filters=self.filters_coef * self.filters, kernel_size=self.kernel_size,
                                      padding='same', activation=self.activation, data_format='channels_last',
                                      groups=1, use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    def call(self, input_, training=True, **kwargs):

        if not isinstance(input_, (np.int32, np.float64, np.float32, np.float16)):
            input_ = cast(input_, 'float16')

        x = self.conv_start(input_)

        conc_list = []

        for i in range(0, self.n_pooling_branches):
            setattr(self, f'x_{i}', getattr(self, f'maxpool_{i}')(x))
            for j in range(self.n_conv_layers):
                setattr(self, f'x_{i}', getattr(self, f'conv_{i, j}')(getattr(self, f'x_{i}')))
            if self.batch_norm_layer:
                setattr(self, f'x_{i}', getattr(self, f'batchnorm_{i}')(getattr(self, f'x_{i}')))
            if self.dropout_layer:
                setattr(self, f'x_{i}', getattr(self, f'dropout_{i}')(getattr(self, f'x_{i}')))
            setattr(self, f'x_{i}', getattr(self, f'convtranspose_{i}')(getattr(self, f'x_{i}')))
            # setattr(self, f'x_{i}', layers.CenterCrop(input_.shape[1], input_.shape[2])(getattr(self, f'x_{i}')))
            conc_list.append(getattr(self, f'x_{i}'))

        concat = self.concatenate(conc_list)
        x = self.conv_end(concat)
        return x

    def get_config(self):
        config = {
            'filters_base': self.filters_base,
            'n_pooling_branches': self.n_pooling_branches,
            'filters_coef': self.filters_coef,
            'n_conv_layers': self.n_conv_layers,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'batch_norm_layer': self.batch_norm_layer,
            'dropout_layer': self.dropout_layer,
        }
        base_config = super(PSPBlock3D, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PretrainedYOLO(Layer):

    def __init__(self, num_classes: int = 5, version: str = "YOLOv4",
                 use_weights: bool = True, save_weights: str = '', **kwargs):
        super(PretrainedYOLO, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.version = version
        self.use_weights = use_weights
        self.save_weights = save_weights
        self.yolo = self.create_yolo(classes=self.num_classes)
        if use_weights:
            self.base_yolo = self.create_yolo()
            self.load_yolo_weights(self.base_yolo, self.save_weights)
            for i, l in enumerate(self.base_yolo.layers):
                layer_weights = l.get_weights()
                if layer_weights:
                    try:
                        self.yolo.layers[i].set_weights(layer_weights)
                    except:
                        print("skipping", self.yolo.layers[i].name)
            del self.base_yolo

    def create_yolo(self, input_size=416, channels=3, classes=80):
        tf.keras.backend.clear_session()  # used to reset layer names
        input_layer = layers.Input([input_size, input_size, channels])
        if self.version == "YOLOv4":
            output_tensors = self.YOLOv4(input_layer, classes)
        else:
            output_tensors = self.YOLOv3(input_layer, classes)
        yolo = tf.keras.Model(input_layer, output_tensors)
        return yolo

    def load_yolo_weights(self, model, weights_file):
        tf.keras.backend.clear_session()  # used to reset layer names
        # load Darknet original weights to TensorFlow model
        if self.version == "YOLOv3":
            range1 = 75
            range2 = [58, 66, 74]
        if self.version == "YOLOv4":
            range1 = 110
            range2 = [93, 101, 109]

        with open(weights_file, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):
                if i > 0:
                    conv_layer_name = 'conv2d_%d' % i
                else:
                    conv_layer_name = 'conv2d'

                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' % j
                else:
                    bn_layer_name = 'batch_normalization'

                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(wf.read()) == 0, 'failed to read all data'

    def convolutional(self, input_layer, filters_shape, downsample=False, activate=True, bn=True,
                      activate_type='leaky'):
        if downsample:
            input_layer = layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                             padding=padding, use_bias=not bn,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate:
            if activate_type == "leaky":
                conv = layers.LeakyReLU(alpha=0.1)(conv)
            elif activate_type == "mish":
                conv = self.mish(conv)
        return conv

    def mish(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

    def residual_block(self, input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        short_cut = input_layer
        conv = self.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1),
                                  activate_type=activate_type)
        conv = self.convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)

        residual_output = short_cut + conv
        return residual_output

    def upsample(self, input_layer):
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    def route_group(self, input_layer, groups, group_id):
        convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
        return convs[group_id]

    def darknet53(self, input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 32))
        input_data = self.convolutional(input_data, (3, 3, 32, 64), downsample=True)

        for i in range(1):
            input_data = self.residual_block(input_data, 64, 32, 64)

        input_data = self.convolutional(input_data, (3, 3, 64, 128), downsample=True)

        for i in range(2):
            input_data = self.residual_block(input_data, 128, 64, 128)

        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = self.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data

    def cspdarknet53(self, input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 32), activate_type="mish")
        input_data = self.convolutional(input_data, (3, 3, 32, 64), downsample=True, activate_type="mish")

        route = input_data
        route = self.convolutional(route, (1, 1, 64, 64), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
        for i in range(1):
            input_data = self.residual_block(input_data, 64, 32, 64, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

        input_data = tf.concat([input_data, route], axis=-1)
        input_data = self.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
        input_data = self.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 128, 64), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
        for i in range(2):
            input_data = self.residual_block(input_data, 64, 64, 64, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 256, 128), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
        for i in range(8):
            input_data = self.residual_block(input_data, 128, 128, 128, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 512, 256), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
        for i in range(8):
            input_data = self.residual_block(input_data, 256, 256, 256, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
        for i in range(4):
            input_data = self.residual_block(input_data, 512, 512, 512, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 1024, 512))
        input_data = self.convolutional(input_data, (3, 3, 512, 1024))
        input_data = self.convolutional(input_data, (1, 1, 1024, 512))

        max_pooling_1 = tf.keras.layers.MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
        max_pooling_2 = tf.keras.layers.MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
        max_pooling_3 = tf.keras.layers.MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 2048, 512))
        input_data = self.convolutional(input_data, (3, 3, 512, 1024))
        input_data = self.convolutional(input_data, (1, 1, 1024, 512))

        return route_1, route_2, input_data

    def YOLOv3(self, input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-53 network, we get three branches
        route_1, route_2, conv = self.darknet53(input_layer)
        # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 512, 1024))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)
        conv = self.convolutional(conv, (1, 1, 768, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv_mobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self.convolutional(conv, (1, 1, 384, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv_sobj_branch = self.convolutional(conv, (3, 3, 128, 256))

        # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
        conv_sbbox = self.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def YOLOv4(self, input_layer, NUM_CLASS):
        route_1, route_2, conv = self.cspdarknet53(input_layer)

        route = conv
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.upsample(conv)
        route_2 = self.convolutional(route_2, (1, 1, 512, 256))
        conv = tf.concat([route_2, conv], axis=-1)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)
        route_1 = self.convolutional(route_1, (1, 1, 256, 128))
        conv = tf.concat([route_1, conv], axis=-1)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))

        route_1 = conv
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = self.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(route_1, (3, 3, 128, 256), downsample=True)
        conv = tf.concat([conv, route_2], axis=-1)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = self.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(route_2, (3, 3, 256, 512), downsample=True)
        conv = tf.concat([conv, route], axis=-1)

        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))

        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = self.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def call(self, input_, training=True, **kwargs):
        return self.yolo(input_)

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
            'version': self.version,
            'use_weights': self.use_weights,
            'save_weights': self.save_weights
        }
        base_config = super(PretrainedYOLO, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return [(None, 52, 52, 3 * (5 + self.num_classes)),
                (None, 26, 26, 3 * (5 + self.num_classes)),
                (None, 13, 13, 3 * (5 + self.num_classes))]


class ConditionalMergeLayer(layers.Layer):
    def __init__(self, mode='Concatenate', **kwargs):
        super(ConditionalMergeLayer, self).__init__(**kwargs)
        self.mode = mode
        pass

    def concatenate(self, input):
        if len(input[0].shape) == len(input[1].shape):
            return layers.Concatenate(axis=-1)([input[0], input[1]])
        elif len(input[0].shape) > len(input[1].shape):
            num = 1
            for i in input[0].shape[1:-1]:
                num *= i
            target_shape = list(input[0].shape[1:-1])
            target_shape.append(input[1].shape[-1])
            x = layers.RepeatVector(num)(input[1])
            x = layers.Reshape(target_shape=target_shape)(x)
            return layers.Concatenate(axis=-1)([input[0], x])
        else:
            num = 1
            for i in input[1].shape[1:-1]:
                num *= i
            target_shape = list(input[1].shape[1:-1])
            target_shape.append(input[0].shape[-1])
            x = layers.RepeatVector(num)(input[0])
            x = layers.Reshape(target_shape=target_shape)(x)
            return layers.Concatenate(axis=-1)([input[1], x])

    def call(self, input, training=True, **kwargs):
        if self.mode == 'Concatenate':
            return self.concatenate(input)

    def get_config(self):
        config = {
            'mode': self.mode,
        }
        base_config = super(ConditionalMergeLayer, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResnetBlock2D(Layer):
    """
    UNET Block2D layer
    n_pooling_branches - defines amt of downsampling/upsampling operations
    filters_coef - defines the multiplication factor for amt of filters in pooling branches
    n_conv_layers - number of conv layers in one downsampling/upsampling segment
    """

    def __init__(self, filters=16, kernel_size=(3, 3), kernel_initializer='RandomNormal', n_conv_layers=2,
                 activation='relu', use_bias=True, use_activation_layer=True, leaky_relu_alpha=0.3,
                 normalization='instance', merge_layer="concatenate", num_resblocks=1,
                 kernel_regularizer=None, **kwargs):

        super(ResnetBlock2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.n_conv_layers = n_conv_layers
        self.activation = activation
        self.normalization = normalization
        self.merge_layer = merge_layer
        self.num_resblocks = num_resblocks
        self.use_bias = use_bias
        self.use_activation_layer = use_activation_layer
        self.leaky_relu_alpha = leaky_relu_alpha
        self.kernel_regularizer = kernel_regularizer

        for i in range(self.num_resblocks):
            for c in range(self.n_conv_layers):
                setattr(self, f"conv_{c + 1}_block_{i + 1}",
                        layers.Conv2D(
                            filters=self.filters, kernel_size=self.kernel_size,
                            activation=None if self.use_activation_layer else self.activation,
                            data_format='channels_last', groups=1, use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer, padding='same',
                            bias_initializer='zeros', kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                            bias_constraint=None))
                if self.normalization:
                    if self.normalization == "batch":
                        setattr(self, f"norm_{c + 1}_block_{i + 1}", layers.BatchNormalization())
                    if self.normalization == "instance":
                        setattr(self, f"norm_{c + 1}_block_{i + 1}", InstanceNormalization())
                if self.use_activation_layer and c + 1 < self.n_conv_layers:
                    if self.activation == "leaky_relu":
                        setattr(self, f"activation_{c + 1}_block_{i + 1}",
                                layers.LeakyReLU(alpha=self.leaky_relu_alpha))
                    if self.activation == "relu":
                        setattr(self, f"activation_{c + 1}_block_{i + 1}", layers.ReLU())
                    if self.activation == "prelu":
                        setattr(self, f"activation_{c + 1}_block_{i + 1}", layers.PReLU())
            if self.merge_layer == "concatenate":
                setattr(self, f"concat_block_{i + 1}", layers.Concatenate(axis=-1))
            if self.merge_layer == "add":
                setattr(self, f"concat_block_{i}", layers.Add())
            if self.merge_layer == "multiply":
                setattr(self, f"concat_block_{i}", layers.Multiply())

    def call(self, input_, training=True, **kwargs):
        y = input_
        for i in range(self.num_resblocks):
            for c in range(self.n_conv_layers):
                x = getattr(self, f"conv_{c + 1}_block_{i + 1}")(y)
                if self.normalization:
                    x = getattr(self, f"norm_{c + 1}_block_{i + 1}")(x)
                if self.use_activation_layer and c + 1 < self.n_conv_layers:
                    x = getattr(self, f"activation_{c + 1}_block_{i + 1}")(x)
            # if i == 0:
            y = getattr(self, f"concat_block_{i + 1}")([y, x])
            # else:
            #     y = getattr(self, f"concat_block_{i + 1}")([y, x])
        return y

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer,
            'n_conv_layers': self.n_conv_layers,
            'activation': self.activation,
            'normalization': self.normalization,
            'merge_layer': self.merge_layer,
            'num_resblocks': self.num_resblocks,
            'use_bias': self.use_bias,
            'use_activation_layer': self.use_activation_layer,
            'kernel_regularizer': self.kernel_regularizer
        }
        base_config = super(ResnetBlock2D, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        if self.merge_layer == "concatenate":
            output_shape = (None, input_shape[1], input_shape[2],
                            self.filters * self.num_resblocks + input_shape[-1])
            return output_shape
        else:
            return input_shape


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tensorflow.keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tensorflow.keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class Transformer(layers.Layer):
    def __init__(self, embed_dim=256, latent_dim=2048, num_heads=8, vocab_size=10000, sequence_length=20, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.pos_emb_encoder = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)
        self.tr_encoder = TransformerEncoder(self.embed_dim, self.latent_dim, self.num_heads)
        self.pos_emb_decoder = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)
        self.tr_decoder = TransformerDecoder(self.embed_dim, self.latent_dim, self.num_heads)
        self.dr_decoder = layers.Dropout(0.5)
        self.fn_decoder = layers.Dense(self.vocab_size, activation="softmax")

    def call(self, input_, training=True, **kwargs):
        x = self.pos_emb_encoder(input_[0])
        encoder_outputs = self.tr_encoder(x)
        x = self.pos_emb_decoder(input_[1])
        x = self.tr_decoder(x, encoder_outputs)
        x = self.dr_decoder(x)
        decoder_outputs = self.fn_decoder(x)
        return decoder_outputs

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
        }
        base_config = super(Transformer, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        output_shape = (None, input_shape[0][1])  # , self.vocab_size
        return output_shape


bert_model_name = ['bert_multi_cased_L-12_H-768_A-12', 'distilbert_multi_cased_L-6_H-768_A-12_32lang',
                   'use-cmlm_multilingual-base-br_100lang', 'large_LaBSE_109lang', "smaller_LaBSE_15lang",
                   'xlm_roberta_multi_cased_L-12_H-768_A-12_53lang', "bert_en_uncased_L-12_H-768_A-12",
                   "bert_en_cased_L-12_H-768_A-12", "small_bert/bert_en_uncased_L-2_H-128_A-2",
                   "small_bert/bert_en_uncased_L-2_H-256_A-4", "small_bert/bert_en_uncased_L-2_H-512_A-8",
                   "small_bert/bert_en_uncased_L-2_H-768_A-12", "small_bert/bert_en_uncased_L-4_H-128_A-2",
                   "small_bert/bert_en_uncased_L-4_H-256_A-4", "small_bert/bert_en_uncased_L-4_H-512_A-8",
                   "small_bert/bert_en_uncased_L-4_H-768_A-12", "small_bert/bert_en_uncased_L-6_H-128_A-2",
                   "small_bert/bert_en_uncased_L-6_H-256_A-4", "small_bert/bert_en_uncased_L-6_H-512_A-8",
                   "small_bert/bert_en_uncased_L-6_H-768_A-12", "small_bert/bert_en_uncased_L-8_H-128_A-2",
                   "small_bert/bert_en_uncased_L-8_H-256_A-4", "small_bert/bert_en_uncased_L-8_H-512_A-8",
                   "small_bert/bert_en_uncased_L-8_H-768_A-12", "small_bert/bert_en_uncased_L-10_H-128_A-2",
                   "small_bert/bert_en_uncased_L-10_H-256_A-4", "small_bert/bert_en_uncased_L-10_H-512_A-8",
                   "small_bert/bert_en_uncased_L-10_H-768_A-12", "small_bert/bert_en_uncased_L-12_H-128_A-2",
                   "small_bert/bert_en_uncased_L-12_H-256_A-4", "small_bert/bert_en_uncased_L-12_H-512_A-8",
                   "small_bert/bert_en_uncased_L-12_H-768_A-12", "albert_en_base", "electra_small", "electra_base",
                   "experts_pubmed", "experts_wiki_books", "talking-heads_base"]


class PretrainedBERT(layers.Layer):
    def __init__(self, model_name: str = '', set_trainable: bool = True, sequence_length: int = 128, **kwargs):
        super(PretrainedBERT, self).__init__(**kwargs)
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.map_name_to_handle = {
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4',
            'distilbert_multi_cased_L-6_H-768_A-12_32lang':
                'https://tfhub.dev/jeongukjae/distilbert_multi_cased_L-6_H-768_A-12/1',
            'use-cmlm_multilingual-base-br_100lang':
                'https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1',
            'large_LaBSE_109lang':
                'https://tfhub.dev/google/LaBSE/2',
            'smaller_LaBSE_15lang':
                'https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1',
            'xlm_roberta_multi_cased_L-12_H-768_A-12_53lang':
                'https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_L-12_H-768_A-12/1',
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_base/2',
            'electra_small':
                'https://tfhub.dev/google/electra_small/2',
            'electra_base':
                'https://tfhub.dev/google/electra_base/2',
            'experts_pubmed':
                'https://tfhub.dev/google/experts/bert/pubmed/2',
            'experts_wiki_books':
                'https://tfhub.dev/google/experts/bert/wiki_books/2',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
        }
        self.map_model_to_preprocess = {
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'distilbert_multi_cased_L-6_H-768_A-12_32lang':
                'https://tfhub.dev/jeongukjae/distilbert_multi_cased_preprocess/2',
            'use-cmlm_multilingual-base-br_100lang':
                'https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2',
            'large_LaBSE_109lang':
                'https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2',
            'smaller_LaBSE_15lang':
                'https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1',
            'xlm_roberta_multi_cased_L-12_H-768_A-12_53lang':
                'https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_preprocess/1',
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'electra_small':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'electra_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_pubmed':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_wiki_books':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        }
        print('self.model_name', self.model_name)
        self.tfhub_handle_encoder = self.map_name_to_handle[self.model_name]
        self.tfhub_handle_preprocess = self.map_model_to_preprocess[self.model_name]
        print('self.tfhub_handle_encoder', self.tfhub_handle_encoder)
        print('self.tfhub_handle_preprocess', self.tfhub_handle_preprocess)
        self.set_trainable = set_trainable
        self.preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        self.encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=self.set_trainable, name='BERT_encoder')
    #     self.net_bert = self.build_classifier_model()
    #     print(self.net_bert.summary())
    #
    # def build_classifier_model(self):
    #     text_input = layers.Input(shape=(), dtype=tf.string, name='text')
    #     preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
    #     encoder_inputs = preprocessing_layer(text_input)
    #     encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    #     outputs = encoder(encoder_inputs)
    #     net = outputs['pooled_output']
    #     return tf.keras.Model(text_input, net)

    def call(self, input_, training=True, **kwargs):
        encoder_inputs = self.preprocessing_layer(input_)
        outputs = self.encoder(encoder_inputs)
        return outputs['pooled_output']
        # #
        # return self.net_bert(input_)

    def get_config(self):
        config = {
            'map_name_to_handle': self.map_name_to_handle,
            'map_model_to_preprocess': self.map_model_to_preprocess,
            'tfhub_handle_encoder': self.tfhub_handle_encoder,
            'tfhub_handle_preprocess': self.tfhub_handle_preprocess,
            'model_name': self.model_name,
            'set_trainable': self.set_trainable,
        }
        base_config = super(PretrainedBERT, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        print('input_shape', input_shape)
        input_ = tf.constant('text', shape=(input_shape[1]), dtype=tf.string)
        print('input_', input_)
        encoder_inputs = self.preprocessing_layer(input_)
        print(encoder_inputs)
        outputs = self.encoder(encoder_inputs)
        print('tf.shape(outputs[pooled_output])', tf.shape(outputs['pooled_output']))
        # outputs = self.net_bert(input_)
        return outputs['pooled_output'].shape



class FNetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super(FNetEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        # Casting the inputs to complex64
        inp_complex = tf.cast(inputs, tf.complex64)
        # Projecting the inputs to the frequency domain using FFT2D and
        # extracting the real part of the output
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class FNetDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(FNetDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


# def create_model():
#     encoder_inputs = keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
#     x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
#     encoder_outputs = FNetEncoder(EMBED_DIM, LATENT_DIM)(x)
#     encoder = keras.Model(encoder_inputs, encoder_outputs)
#     decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
#     encoded_seq_inputs = keras.Input(
#         shape=(None, EMBED_DIM), name="decoder_state_inputs"
#     )
#     x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
#     x = FNetDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x, encoded_seq_inputs)
#     x = layers.Dropout(0.5)(x)
#     decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
#     decoder = keras.Model(
#         [decoder_inputs, encoded_seq_inputs], decoder_outputs, name="outputs"
#     )
#     decoder_outputs = decoder([decoder_inputs, encoder_outputs])
#     fnet = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="fnet")
#     return fnet



if __name__ == "__main__":
    # input = tensorflow.keras.layers.Input(shape=(32, 32, 3))
    # x = YOLOResBlock(32, 2)(input)
    # print(x)
    # block_type = 'YOLOResBlock'
    # x = YOLOResBlock(**{'mode': "YOLOv5", 'filters': 32, "num_resblocks": 5, "activation": 'Swish',
    #                     "use_bias": False, "include_head": True, "include_add": True,
    #                     "all_narrow": True})
    # x = YOLOConvBlock(**{'mode': "YOLOv5", "filters": 64, "num_conv": 5, 'activation': 'Swish'})
    # x = UNETBlock2D(filters_base=16, n_pooling_branches=3, filters_coef=1, n_conv_layers=1, activation='relu',
    #                 kernel_size=(3, 3), batch_norm_layer=True,
    #                 dropout_layer=True, dropout_rate=0.1)
    # x = PSPBlock2D(filters_base=32, n_pooling_branches=3, filters_coef=1, n_conv_layers=2, activation='relu',
    #                kernel_size=(3, 3), batch_norm_layer = True, dropout_layer = True, dropout_rate = 0.1)
    # x = PSPBlock3D(filters_base=32, n_pooling_branches=3, filters_coef=1, n_conv_layers=1, activation='relu',
    #                kernel_size=(3, 3, 3), batch_norm_layer=True, dropout_layer=True, dropout_rate=0.1)
    # x = PSPBlock1D(filters_base=32, n_pooling_branches=3, filters_coef=1, n_conv_layers=2, activation='relu',
    #                kernel_size=5, batch_norm_layer = True, dropout_layer = True, dropout_rate = 0.1)
    # x = UNETBlock1D(filters_base=16, n_pooling_branches=3, filters_coef=1, n_conv_layers=2, activation='relu',
    #                 kernel_size=5, batch_norm_layer=True,
    #                 dropout_layer=True, dropout_rate=0.1)
    # x = UNETBlock3D(filters_base=16, n_pooling_branches=3, filters_coef=1, n_conv_layers=1, activation='relu',
    #                 kernel_size=(3,3,3), batch_norm_layer=True,
    #                 dropout_layer=True, dropout_rate=0.1)
    # x = OnlyYolo(use_weights=True, num_classes=1, yolo_version='v4')
    # aa = x.build((64, 64, 3))
    # x.summary()
    # tf.keras.utils.plot_model(
    #     aa, to_file='C:\PycharmProjects\\terra_gui\\test_example\\model.png', show_shapes=True, show_dtype=False,
    #     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
    #     layer_range=None, show_layer_activations=False)
    # x = InstanceNormalization()
    # print(x.compute_output_shape(input_shape=(None, 100)))

    # params = {
    #     'filters_base': 16, 'n_pooling_branches': 5, 'activation': None, 'normalization': 'instance',
    #     'dropout_layer': False, 'leaky_relu_layer': True, 'use_activation_layer': False,
    #     'maxpooling': False, 'upsampling': False, 'filters_coef': 1, 'n_conv_layers': 4,
    #     'use_bias': False, 'kernel_size': [3, 3], 'kernel_initializer': 'random_normal',
    #     'dropout_rate': 0.1, 'leaky_relu_alpha': 0.3, 'name': 'UNETBlock2D_3'
    # }
    params = {'model_name': 'small_bert/bert_en_uncased_L-10_H-512_A-8', 'set_trainable': True}

    # params = {
    #     'filters': 256, 'num_resblocks': 9, 'n_conv_layers': 2, 'use_activation_layer': True,
    #     'activation': 'relu', 'kernel_size': (3, 3), 'kernel_initializer': 'glorot_uniform',
    #     'normalization': 'instance', "merge_layer": 'concatenate', "use_bias": True
    # }
    # # for i in range(5):
    input_shape = (20,1)

    text_input = tensorflow.keras.Input(shape=(), dtype=tf.string)
    layer = PretrainedBERT(**params)
    # print(input)
    x = layer(text_input)
    print(x.shape)
    model = tf.keras.Model(text_input, x)
    model.summary()
    text = np.array([["раз one two three 123125 . !"], ["text one two three 123125 data one. !"]])
    print(text.shape,text[0])
    pred = model(text)
    print(pred.shape)
    print(pred)
    print('layer.compute_output_shape', layer.compute_output_shape(input_shape=input_shape))
    pass
