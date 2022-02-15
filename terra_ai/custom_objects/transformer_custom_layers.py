import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow.keras import layers


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


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
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
        self.dense_proj = tf.keras.Sequential(
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
    def __init__(self, embed_dim=256, latent_dim=2048, num_heads=8, vocab_size_enc=10000, vocab_size_dec=10000,
                 enc_seq_length=20, dec_seq_length=20, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.vocab_size_enc = vocab_size_enc
        self.vocab_size_dec = vocab_size_dec
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length
        self.pos_emb_encoder = PositionalEmbedding(self.enc_seq_length, self.vocab_size_enc, self.embed_dim)
        self.tr_encoder = TransformerEncoder(self.embed_dim, self.latent_dim, self.num_heads)
        self.pos_emb_decoder = PositionalEmbedding(self.dec_seq_length, self.vocab_size_dec, self.embed_dim)
        self.tr_decoder = TransformerDecoder(self.embed_dim, self.latent_dim, self.num_heads)
        self.dr_decoder = layers.Dropout(0.5)
        self.fn_decoder = layers.Dense(self.vocab_size_dec, activation="softmax")

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
            'vocab_size_enc': self.vocab_size_enc,
            'vocab_size_dec': self.vocab_size_dec,
            'enc_seq_length': self.enc_seq_length,
            'dec_seq_length': self.dec_seq_length,
        }
        base_config = super(Transformer, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        output_shape = (None, input_shape[1][1]) #, self.vocab_size_dec
        return output_shape
        # input_0 = tf.keras.Input(shape=(input_shape[0][1],), dtype=tf.int64)
        # input_1 = tf.keras.Input(shape=(input_shape[1][1],), dtype=tf.int64)
        # outputs = self.call([input_0, input_1])
        # return outputs.shape


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

class FNetTransformer(layers.Layer):
    def __init__(self, embed_dim=256, latent_dim=2048, num_heads=8, vocab_size_enc=10000, vocab_size_dec=10000,
                 enc_seq_length=20, dec_seq_length=20, **kwargs):
        super(FNetTransformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.vocab_size_enc = vocab_size_enc
        self.vocab_size_dec = vocab_size_dec
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length
        self.pos_emb_encoder = PositionalEmbedding(self.enc_seq_length, self.vocab_size_enc, self.embed_dim)
        self.tr_encoder = FNetEncoder(self.embed_dim, self.latent_dim)
        self.pos_emb_decoder = PositionalEmbedding(self.dec_seq_length, self.vocab_size_dec, self.embed_dim)
        self.tr_decoder = FNetDecoder(self.embed_dim, self.latent_dim, self.num_heads)
        self.dr_decoder = layers.Dropout(0.5)
        self.fn_decoder = layers.Dense(self.vocab_size_dec, activation="softmax")

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
            'vocab_size_enc': self.vocab_size_enc,
            'vocab_size_dec': self.vocab_size_dec,
            'enc_seq_length': self.enc_seq_length,
            'dec_seq_length': self.dec_seq_length,
        }
        base_config = super(FNetTransformer, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        # tf.print('input_shape', input_shape)
        output_shape = (None, input_shape[1][1]) #, self.vocab_size_dec
        return output_shape
        # input_0 = tf.keras.Input(shape=(input_shape[0][1],), dtype=tf.int64)
        # input_1 = tf.keras.Input(shape=(input_shape[1][1],), dtype=tf.int64)
        # outputs = self.call([input_0, input_1])
        # return outputs.shape


class BERT(layers.Layer):
    def __init__(self, embed_dim, num_layers, num_heads, max_len, vocab_size, ff_dim, **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.ff_dim = ff_dim

        self.word_embeddings = layers.Embedding(self.vocab_size, self.embed_dim, name="word_embedding")
        self.position_embeddings = layers.Embedding(
            input_dim=self.max_len,
            output_dim=self.embed_dim,
            weights=[self.get_pos_encoding_matrix(self.max_len, self.embed_dim)],
            name="position_embedding")
        self.mlm_output = layers.Dense(self.vocab_size, name="mlm_cls", activation="softmax")


    def bert_module(self, query, key, value, i):
        # Multi headed self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            name="encoder_{}/multiheadattention".format(i),
        )(query, key, value)
        attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
            attention_output
        )
        attention_output = layers.LayerNormalization(
            epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
        )(query + attention_output)

        # Feed-forward layer
        ffn = tf.keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation="relu"),
                layers.Dense(self.embed_dim),
            ],
            name="encoder_{}/ffn".format(i),
        )
        ffn_output = ffn(attention_output)
        ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
            ffn_output
        )
        sequence_output = layers.LayerNormalization(
            epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
        )(attention_output + ffn_output)
        return sequence_output


    def get_pos_encoding_matrix(self, max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc


    # class MaskedLanguageModel(tf.keras.Model):
    #     def train_step(self, inputs):
    #         if len(inputs) == 3:
    #             features, labels, sample_weight = inputs
    #         else:
    #             features, labels = inputs
    #             sample_weight = None
    #
    #         with tf.GradientTape() as tape:
    #             predictions = self(features, training=True)
    #             loss = self.loss_fn(labels, predictions, sample_weight=sample_weight)
    #
    #         # Compute gradients
    #         trainable_vars = self.trainable_variables
    #         gradients = tape.gradient(loss, trainable_vars)
    #
    #         # Update weights
    #         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #
    #         # Compute our own metrics
    #         self.loss_tracker.update_state(loss, sample_weight=sample_weight)
    #
    #         # Return a dict mapping metric names to current value
    #         return {"loss": self.loss_tracker.result()}
    #
    #     @property
    #     def metrics(self):
    #         # We list our `Metric` objects here so that `reset_states()` can be
    #         # called automatically at the start of each epoch
    #         # or at the start of `evaluate()`.
    #         # If you don't implement this property, you have to call
    #         # `reset_states()` yourself at the time of your choosing.
    #         return [self.loss_tracker]


    def call(self, input_, training=True, **kwargs):
        # inputs = layers.Input((self.max_len,), dtype=tf.int64)

        word_embeddings = self.word_embeddings(input_)
        position_embeddings = self.position_embeddings(tf.range(start=0, limit=self.max_len, delta=1))
        embeddings = word_embeddings + position_embeddings

        encoder_output = embeddings
        for i in range(self.num_layers):
            encoder_output = self.bert_module(encoder_output, encoder_output, encoder_output, i)

        mlm_output = self.mlm_output(encoder_output)

        return mlm_output

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'ff_dim': self.ff_dim,
        }
        base_config = super(BERT, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        # print('input_shape', ((1,)+input_shape))
        input_ = tf.keras.Input(shape=(input_shape), dtype=tf.int64)
        # input_ = tf.constant(1, shape=(((1,)+input_shape)), dtype=tf.int64)
        # print('input_', input_)
        outputs = self.call(input_)
        # print('tf.shape(outputs)', tf.shape(outputs))
        return outputs.shape


if __name__ == "__main__":

    #FNetTransformer
    params = {'embed_dim':256, 'latent_dim':512, 'num_heads':8, 'vocab_size_enc':10000, 'vocab_size_dec':10000,
                 'enc_seq_length':40, 'dec_seq_length':40}
    input_shape = ((None, 40,),(None, 40,))
    layer = FNetTransformer(**params)
    print('layer.compute_output_shape', layer.compute_output_shape(input_shape=input_shape))
    pass

    # #Transformer
    # params = {'embed_dim':256, 'latent_dim':512, 'num_heads':8, 'vocab_size_enc':10000, 'vocab_size_dec':20000,
    #              'enc_seq_length':50, 'dec_seq_length':50}
    # input_shape = ((None, 50,),(None, 50,))
    # layer = Transformer(**params)
    # print('layer.compute_output_shape', layer.compute_output_shape(input_shape=input_shape))
    # pass

    # #BERT
    # params = {'embed_dim': 128, 'num_layers': 1, 'num_heads': 8, 'max_len': 256, 'vocab_size': 30000, 'ff_dim': 128}
    # input_shape = (256,)
    # text_input = tf.keras.Input(shape=(input_shape), dtype=tf.int64)
    # layer = Bert(**params)
    # x = layer(text_input)
    # print(x.shape)
    # model = tf.keras.Model(text_input, x)
    # model.summary()
    # # text = np.ones((1, 256))
    # # pred = model(text)
    # # print(pred.shape)
    # # print(pred)
    # print('layer.compute_output_shape', layer.compute_output_shape(input_shape=input_shape))
    # pass