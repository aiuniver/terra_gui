from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import layers


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

        epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)

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


class ConditionalMergeLayer(layers.Layer):
    def __init__(self, mode='Concatenate', **kwargs):
        super(ConditionalMergeLayer, self).__init__(**kwargs)
        self.mode = mode  # Concatenate Multiply
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

    # def multiply(self, input):
    #     if len(input[0].shape[1:]) == len(input[1].shape[1:]):
    #         if input[0].shape[-1] >= input[1].shape[-1]:
    #             cond_input = input[1]
    #             second_input = layers.Flatten()(input[0])
    #         else:
    #             cond_input = input[0]
    #             second_input = layers.Flatten()(input[1])
    #     else:
    #         if len(input[0].shape[1:]) > len(input[1].shape[1:]):
    #             cond_input = input[1]
    #             second_input = layers.Flatten()(input[0])
    #         else:
    #             cond_input = input[0]
    #             second_input = layers.Flatten()(input[1])
    #     # if tf.reduce_sum(input[0]) == 1 * input[0].shape[0]:
    #     #     cond_input = input[0]
    #     #     second_input = layers.Flatten()(input[1])
    #     # else:
    #     #     cond_input = input[1]
    #     #     second_input = layers.Flatten()(input[0])
    #     labels = tf.expand_dims(tf.argmax(cond_input, axis=-1), axis=-1)
    #     labels = tf.cast(labels, dtype='float32')
    #     print('cond_input.shape, second_input.shape', cond_input.shape, second_input.shape, labels.shape)
    #     # input_dim = cond_input.shape[-1], output_dim = second_input.shape[-1]
    #     x = layers.Embedding(input_dim=cond_input.shape[-1], output_dim=second_input.shape[-1])(labels)
    #     print(x.shape)
    #     label_embedding = layers.Flatten()(x)
    #     print('label_embedding', label_embedding.shape)
    #     Multiply = layers.Multiply()([second_input, label_embedding])
    #     print('Multiply', Multiply.shape)
    #     return Multiply
    #     # elif len(input[0].shape) > len(input[1].shape):
    #     #     num = 1
    #     #     for i in input[0].shape[1:-1]:
    #     #         num *= i
    #     #     target_shape = list(input[0].shape[1:-1])
    #     #     target_shape.append(input[1].shape[-1])
    #     #     x = layers.RepeatVector(num)(input[1])
    #     #     x = layers.Reshape(target_shape=target_shape)(x)
    #     #     return layers.Concatenate(axis=-1)([input[0], x])
    #     # else:
    #     #     num = 1
    #     #     for i in input[1].shape[1:-1]:
    #     #         num *= i
    #     #     target_shape = list(input[1].shape[1:-1])
    #     #     target_shape.append(input[0].shape[-1])
    #     #     x = layers.RepeatVector(num)(input[0])
    #     #     x = layers.Reshape(target_shape=target_shape)(x)
    #     #     return layers.Concatenate(axis=-1)([input[1], x])

    def call(self, input, training=True, **kwargs):
        if self.mode == 'Concatenate':
            return self.concatenate(input)
        # if self.mode == 'Multiply':
        #     print(input)
        #     return self.multiply(input)

    def get_config(self):
        config = {
            'mode': self.mode,
        }
        base_config = super(ConditionalMergeLayer, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # def compute_output_shape(self, input_shape):
    #     if self.mode == 'Concatenate':
    #         if len(input_shape[0][1:]) == len(input_shape[1][1:]):
    #             return None, input_shape[0][-1] + input_shape[1][-1]
    #         elif len(input_shape[0][1:]) > len(input_shape[1][1:]):
    #             shape = [None]
    #             shape.extend(input_shape[0][1:-1])
    #             shape.append(input_shape[0][-1] + input_shape[1][-1])
    #             return tuple(shape)
    #         else:
    #             shape = [None]
    #             shape.extend(input_shape[1][1:-1])
    #             shape.append(input_shape[1][-1] + input_shape[0][-1])
    #             return tuple(shape)
    #     if self.mode == 'Multiply':
    #         if len(input_shape[0][1:]) == len(input_shape[1][1:]):
    #             if input_shape[0][-1] >= input_shape[1][-1]:
    #                 max_inp = input_shape[0][-1]
    #             else:
    #                 max_inp = input_shape[1][-1]
    #             return None, max_inp
    #         else:
    #             print(input_shape)
    #             if len(input_shape[0][1:]) > len(input_shape[1][1:]):
    #                 return None, np.prod(input_shape[0][1:]).astype('int')
    #             else:
    #                 return None, np.prod(input_shape[1][1:]).astype('int')


class VAEDiscriminatorBlock(Layer):

    def __init__(self, conv_filters=128, dense_units=1024, use_bias=True, leaky_relu_alpha=0.3, bn_momentum=0.99,
                 **kwargs):
        super(VAEDiscriminatorBlock, self).__init__(**kwargs)
        self.conv_filters = conv_filters
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dense_units = dense_units
        self.use_bias = use_bias
        self.bn_momentum = bn_momentum
        # self.batch_size = batch_size

        self.conv_mean = layers.Conv2D(
            filters=self.conv_filters, kernel_size=1, strides=1, padding='same', use_bias=self.use_bias)
        self.conv_logvar = layers.Conv2D(
            filters=self.conv_filters, kernel_size=1, strides=1, padding='same', use_bias=self.use_bias)
        self.dense = layers.Dense(units=self.dense_units, use_bias=self.use_bias)
        self.bn = layers.BatchNormalization(momentum=self.bn_momentum)
        self.leaky = layers.LeakyReLU(alpha=self.leaky_relu_alpha)
        self.output_dense = layers.Dense(units=1, activation='sigmoid', name="output_dense_VAEDiscriminatorBlock")

    def call(self, inputs, training=False, **kwargs):
        input_size = inputs.shape[1] * inputs.shape[2]
        mean = self.conv_mean(inputs)
        mean = layers.Reshape(target_shape=(self.conv_filters * input_size,), name="mean_VAEDiscriminatorBlock")(mean)
        logvar = self.conv_logvar(inputs)
        logvar = layers.Reshape(target_shape=(self.conv_filters * input_size,), name="logvar_VAEDiscriminatorBlock")(
            logvar)
        # if mean.shape[0]:
        #     noise = tf.random.normal(mean.shape)
        # else:
        #     noise = tf.expand_dims(tf.random.normal(mean.shape[1:]), axis=0)
        # print('mean.shape', mean.shape)
        noise = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
        z = K.exp(0.5 * logvar) * noise + mean
        # print()
        # if training:
        #     noise = tf.random.normal(mean.shape)
        #     z = tf.exp(0.5 * logvar) * noise + mean
        # else:
        #     noise = tf.random.normal((self.conv_filters * input_size,))
        #     # noise = tf.fill(mean.shape, 1.)
        #     z = tf.exp(0.5 * logvar) * noise + mean
        # print('z', z.shape)

        x = self.dense(z)
        x = self.bn(x)
        x = self.leaky(x)
        x = self.output_dense(x)

        return x, mean, logvar

    def get_config(self):
        config = {
            'conv_filters': self.conv_filters,
            'dense_units': self.dense_units,
            # 'batch_size': self.batch_size,
            'use_bias': self.use_bias,
            'leaky_relu_alpha': self.leaky_relu_alpha,
            'bn_momentum': self.bn_momentum
        }
        base_config = super(VAEDiscriminatorBlock, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        # return [(None, 1),
        #         (None, self.conv_filters * input_shape[1] * input_shape[2]),
        #         (None, self.conv_filters * input_shape[1] * input_shape[2])]
        return None, 1


if __name__ == "__main__":
    pass
