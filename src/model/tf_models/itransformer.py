# src/model/tf_models/itransformer.py
import tensorflow as tf
from tensorflow.keras import layers, Sequential

class InstanceNormSeries(layers.Layer):
    def __init__(self, eps=1e-5, **kw):
        super().__init__(**kw)
        self.eps = eps
    def call(self, x):
        m = tf.reduce_mean(x, axis=-1, keepdims=True)
        v = tf.reduce_mean(tf.square(x - m), axis=-1, keepdims=True)
        return (x - m) / tf.sqrt(v + self.eps)

class SeriesEmbedding(layers.Layer):
    def __init__(self, d_model: int, **kw):
        super().__init__(**kw)
        self.proj = layers.Dense(int(d_model))
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    def call(self, x):
        return self.norm(self.proj(x))

class ChannelSinusoidalPE(layers.Layer):
    def __init__(self, d_model: int, **kw):
        super().__init__(**kw)
        self.d_model = int(d_model)
    def call(self, z):
        C = tf.shape(z)[1]
        d = self.d_model
        half = d // 2
        pos = tf.cast(tf.range(C)[:, None], tf.float32)
        div = tf.pow(10000.0, -tf.range(half, dtype=tf.float32) / float(half))
        ang = pos * div[None, :]
        pe = tf.reshape(tf.stack([tf.sin(ang), tf.cos(ang)], axis=-1), (C, -1))
        if d % 2 == 1:
            pe = tf.pad(pe, [[0, 0], [0, 1]])
        pe = pe[None, :, :]
        return z + pe

class ChannelTransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads=4, dff=None, dropout=0.0, **kw):
        super().__init__(**kw)
        self.d_model   = int(d_model)
        self.num_heads = max(1, min(int(num_heads), self.d_model))
        self.key_dim   = max(1, self.d_model // self.num_heads)
        self.dropout   = float(dropout)

        self.proj_in = layers.Dense(self.d_model)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)

        try:
            self.mha = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.key_dim,
                output_shape=self.d_model, dropout=self.dropout
            )
            self._needs_proj_out = False
        except TypeError:
            self.mha = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout
            )
            self._needs_proj_out = True
            self.proj_out = layers.Dense(self.d_model)

        if dff is None:
            dff_hidden = max(64, 4 * self.d_model)
        else:
            dff_hidden = int(dff)

        self.do1 = layers.Dropout(self.dropout)
        self.ffn = Sequential([
            layers.Dense(dff_hidden, activation="relu"),
            layers.Dense(self.d_model),
        ])
        self.do2 = layers.Dropout(self.dropout)

    def call(self, x, training=False):
        if x.shape[-1] != self.d_model:
            x = self.proj_in(x)
        qkv = self.ln1(x)
        y = self.mha(qkv, qkv, qkv, training=training)
        if getattr(self, "_needs_proj_out", False):
            y = self.proj_out(y)
        x = x + self.do1(y, training=training)
        y2 = self.ffn(self.ln2(x))
        x  = x + self.do2(y2, training=training)
        return x
