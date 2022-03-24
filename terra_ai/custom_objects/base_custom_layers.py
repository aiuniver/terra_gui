import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow import cast
from tensorflow.keras import layers

from terra_ai.custom_objects.normalization_custom_layers import InstanceNormalization


class CONVBlock(Layer):
    """Conv block layer """

    def __init__(
            self, n_conv_layers=2, filters=16, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='RandomNormal',
            dilation=(1, 1), padding='same', activation='relu', transpose=False, use_bias=True,
            use_activation_layer=False, leaky_relu_layer=True, leaky_relu_alpha=0.3,
            normalization='batch', dropout_layer=True, dropout_rate=0.1, kernel_regularizer=None,
            layers_seq_config: str = 'conv_bn_lrelu_drop_conv_bn_lrelu_drop', bn_momentum=0.99,
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
        self.bn_momentum = bn_momentum

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
                setattr(self, f"norm_{i}", layers.BatchNormalization(axis=-1, momentum=self.bn_momentum))
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
                 kernel_regularizer=None, bn_momentum=0.99, prelu_shared_axes=None, **kwargs):

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
        self.bn_momentum = bn_momentum
        self.prelu_shared_axes = prelu_shared_axes

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
                        setattr(self, f"norm_{c + 1}_block_{i + 1}",
                                layers.BatchNormalization(momentum=self.bn_momentum))
                    if self.normalization == "instance":
                        setattr(self, f"norm_{c + 1}_block_{i + 1}", InstanceNormalization())
                if self.use_activation_layer and c + 1 < self.n_conv_layers:
                    if self.activation == "leaky_relu":
                        setattr(self, f"activation_{c + 1}_block_{i + 1}",
                                layers.LeakyReLU(alpha=self.leaky_relu_alpha))
                    if self.activation == "relu":
                        setattr(self, f"activation_{c + 1}_block_{i + 1}", layers.ReLU())
                    if self.activation == "prelu":
                        setattr(self, f"activation_{c + 1}_block_{i + 1}",
                                layers.PReLU(shared_axes=self.prelu_shared_axes))
            if self.merge_layer == "concatenate":
                setattr(self, f"concat_block_{i + 1}", layers.Concatenate(axis=-1))
            if self.merge_layer == "add":
                setattr(self, f"concat_block_{i + 1}", layers.Add())
            if self.merge_layer == "multiply":
                setattr(self, f"concat_block_{i + 1}", layers.Multiply())

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


if __name__ == "__main__":
    pass
