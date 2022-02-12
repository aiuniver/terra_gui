import tensorflow as tf
import tensorflow
from tensorflow.python.layers.base import Layer
from tensorflow.keras.layers import BatchNormalization

from terra_ai.custom_objects.normalization_custom_layers import Mish


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


if __name__ == "__main__":
    pass
