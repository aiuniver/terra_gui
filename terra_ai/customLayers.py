from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf


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

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
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

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
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
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomUNETBlock(Layer):
    """
    Unet block layer

    # Arguments
        filters: Default: 32
        activation: Default: 'relu', or any possible activation.
        """

    def __init__(self,
                 filters=32,
                 activation='relu',
                 **kwargs):
        super(CustomUNETBlock, self).__init__(**kwargs)
        self.filters = filters
        self.activation = activation
        self.x_1 = layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 activation=self.activation,
                                 data_format='channels_last', dilation_rate=(1, 1), groups=1, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                 bias_constraint=None)
        self.x_2 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                             gamma_constraint=None)
        self.x_3 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')
        self.x_4 = layers.Conv2D(filters=self.filters * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 activation=self.activation,
                                 data_format='channels_last', dilation_rate=(1, 1), groups=1, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                 bias_constraint=None)
        self.x_5 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                             gamma_constraint=None)
        self.x_6 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')
        self.x_7 = layers.Conv2D(filters=self.filters * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 activation=self.activation,
                                 data_format='channels_last', dilation_rate=(1, 1), groups=1, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                 bias_constraint=None)
        self.x_8 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                             gamma_constraint=None)
        self.x_9 = layers.Conv2D(filters=self.filters * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 activation=self.activation,
                                 data_format='channels_last', dilation_rate=(1, 1), groups=1, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                 bias_constraint=None)
        self.x_10 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones',
                                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                              gamma_constraint=None)
        self.x_11 = layers.Conv2DTranspose(filters=self.filters * 2, kernel_size=(2, 2), strides=(2, 2), padding='same',
                                           activation=self.activation, output_padding=None, data_format='channels_last',
                                           dilation_rate=(1, 1), use_bias=True, kernel_initializer='glorot_uniform',
                                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                           activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        self.x_12 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones',
                                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                              gamma_constraint=None)
        self.x_13 = layers.Concatenate(axis=-1)
        self.x_14 = layers.Conv2D(filters=self.filters * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                  activation=self.activation,
                                  data_format='channels_last', dilation_rate=(1, 1), groups=1, use_bias=True,
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)
        self.x_15 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones',
                                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                              gamma_constraint=None)
        self.x_16 = layers.Conv2DTranspose(filters=self.filters, kernel_size=(2, 2), strides=(2, 2), padding='same',
                                           activation=self.activation,
                                           output_padding=None, data_format='channels_last', dilation_rate=(1, 1),
                                           use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                           kernel_constraint=None, bias_constraint=None)
        self.x_17 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones',
                                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                              gamma_constraint=None)
        self.x_18 = layers.Concatenate(axis=-1)
        self.x_19 = layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                  activation=self.activation,
                                  data_format='channels_last', dilation_rate=(1, 1), groups=1, use_bias=True,
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)

    def call(self, inputs, training=True):
        x_1 = self.x_1(inputs)
        x_2 = self.x_2(x_1)
        x_3 = self.x_3(x_2)
        x_4 = self.x_4(x_3)
        x_5 = self.x_5(x_4)
        x_6 = self.x_6(x_5)
        x_7 = self.x_7(x_6)
        x_8 = self.x_8(x_7)
        x_9 = self.x_9(x_8)
        x_10 = self.x_10(x_9)
        x_11 = self.x_11(x_10)
        x_12 = self.x_12(x_11)
        x_13 = self.x_13([x_12, x_5])
        x_14 = self.x_14(x_13)
        x_15 = self.x_15(x_14)
        x_16 = self.x_16(x_15)
        x_17 = self.x_17(x_16)
        x_18 = self.x_18([x_17, x_2])
        x_19 = self.x_19(x_18)
        return x_19

    def get_config(self):
        config = {
            'filters': self.filters,
            'activation': self.activation,
        }
        base_config = super(CustomUNETBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VAEBlock(Layer):
    '''
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
    '''

    def __init__(self, latent_size=32, latent_regularizer='vae', beta=5.,
                 capacity=128., randomSample=True, roll_up=True, **kwargs):
        '''
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
        randomSample : bool
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
        '''
        super(VAEBlock, self).__init__(name='vaeblock', **kwargs)
        # sampling
        self.reg = latent_regularizer
        self.beta = beta
        self.capacity = capacity
        self.random = randomSample
        # variational encoder
        self.latent_size = latent_size
        self.roll_up = roll_up
        self.conv_mean = layers.Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                                       padding='same')
        self.gla_mean = layers.GlobalAveragePooling2D()
        self.conv_stddev = layers.Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                                         padding='same')
        self.gla_stddev = layers.GlobalAveragePooling2D()
        self.inter_dense = layers.Dense(8 * self.latent_size, activation='relu')
        self.dense_mean = layers.Dense(self.latent_size)
        self.dense_stddev = layers.Dense(self.latent_size)

    def call(self, inputs):
        # variational encoder output (distributions)
        if len(K.shape(inputs)) == 4 or len(K.shape(inputs)) == 4:
            mean = self.conv_mean(inputs)
            stddev = self.conv_stddev(inputs)
            if self.roll_up:
                mean = self.gla_mean(mean)
                stddev = self.gla_stddev(stddev)

        elif len(K.shape(inputs)) == 2 or len(K.shape(inputs)) == 2:
            inter = self.inter_dense(inputs)
            mean = self.dense_mean(inter)
            stddev = self.dense_stddev(inter)
        else:
            raise Exception(
                'input shape VAEBlock is not a vector [batchSize, intermediate_dim] or [batchSize, width, heigth, ch]')
        if self.reg:
            # kl divergence:
            latent_loss = -0.5 * tf.reduce_sum(1 + stddev
                                               - K.square(mean)
                                               - K.exp(stddev), axis=-1)
            if self.reg == 'bvae':
                # use beta to force less usage of vector space:
                # also try to use <capacity> dimensions of the space:
                latent_loss = self.beta * K.abs(latent_loss - self.capacity / self.latent_size)
            self.add_loss(tf.reduce_mean(latent_loss))

        epsilon = K.random_normal(shape=K.shape(mean),
                                  mean=0., stddev=1.)

        if self.random:
            # 'reparameterization trick':
            return mean + K.exp(stddev / 2) * epsilon
        else:  # do not perform random sampling, simply grab the impulse value
            return mean + 0 * stddev  # Keras needs the *0 so the gradinent is not None

    # def compute_output_shape(self, input_shape):
    #     return tf.shape(input_shape)[0]

    def get_config(self):
        config = {
            'filters': self.filters,
            'activation': self.activation,
            'latent_regularizer': self.reg,
            'beta': self.beta,
            'capacity': self.capacity,
            'randomSample': self.random,
            'latent_size': self.latent_size,
            'roll_up': self.roll_up,
        }
        base_config = super(VAEBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
