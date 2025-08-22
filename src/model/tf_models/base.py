import tensorflow as tf
from tensorflow.keras import layers
from .layer import get_base_layer
from .layer import InstanceNormSeries, SeriesEmbedding, ChannelSinusoidalPE

class BaseModel(tf.keras.Model):
    def __init__(self, parameters, n_features_in=7, n_features_out=1):
        super().__init__()
        self.parameters = parameters
        self.model = parameters['model']['name']
        self.n_layers = parameters['model']['params']['layers']
        self.n_units = parameters['model']['params']['units']
        self.dropout = parameters['model']['params']['dropout']
        self.pred_len = parameters['dataset']['params']['pred_len']
        self.seq_len = parameters['dataset']['params']['seq_len']

        self.BASE_LAYER = get_base_layer(self.model)

        self.n_outputs = n_features_out * self.pred_len
        self.reshape_layer = layers.Reshape((self.seq_len, n_features_in), name='inputs_reshaped')

        if self.model == 'dense':
            self.hidden_layers = [self.BASE_LAYER(self.n_units, name=f'layer_{i}') for i in range(self.n_layers)]
            self.reshape_layer = layers.Reshape((self.seq_len * n_features_in,), name='inputs_reshaped')

        elif self.model == 'lstm':
            self.hidden_layers = [
                self.BASE_LAYER(self.n_units, name=f'layer_{i}', return_sequences=True if i < (self.n_layers - 1) else False)
                for i in range(self.n_layers)
            ]
            self.reshape_layer = layers.Reshape((self.seq_len, n_features_in), name='inputs_reshaped')

        elif self.model in ('cnn', 'tcn', 'transformer', 'itransformer'):
            self.reshape_layer = layers.Reshape((self.seq_len, n_features_in), name='inputs_reshaped')

            if self.model == 'cnn':
                ks = int(self.parameters['model']['params'].get('kernel_size', 3))
                self.hidden_layers = [
                    self.BASE_LAYER(self.n_units, name=f'layer_{i}', kernel_size=ks)
                    for i in range(self.n_layers)
                ]

            elif self.model == 'tcn':
                ks   = int(self.parameters['model']['params'].get('kernel_size', 3))
                base = int(self.parameters['model']['params'].get('dilation_base', 2))
                self.hidden_layers = []
                for i in range(self.n_layers):
                    d = max(1, base ** i)
                    self.hidden_layers.append(
                        self.BASE_LAYER(filters=self.n_units, kernel_size=ks, dilation_rate=d, dropout=self.dropout, name=f'layer_{i}')
                    )

            elif self.model in ('transformer', 'itransformer'):
                heads    = int(self.parameters['model']['params'].get('num_heads', max(1, min(8, self.n_units // 32))))
                dff_mult = float(self.parameters['model']['params'].get('dff_mult', 4.0))
                dff      = max(64, int(dff_mult * self.n_units))
                self.hidden_layers = [
                    self.BASE_LAYER(d_model=self.n_units, num_heads=heads, dff=dff, dropout=self.dropout, name=f'layer_{i}')
                    for i in range(self.n_layers)
                ]

            if self.model == 'itransformer':
                self._instnorm_series = InstanceNormSeries(name='instancenorm_series')
                self._series_embed = SeriesEmbedding(d_model=self.n_units, name='series_embed')
                self._channel_pe   = ChannelSinusoidalPE(d_model=self.n_units, name='channel_pe')
                self.label_idxs = [int(i) for i in parameters.get('dataset', {}).get('label_idxs', [])] or None

            self.temporal_pool = layers.GlobalAveragePooling1D(name='temporal_pool')

        self.dropout_hidden_layers = [layers.Dropout(self.dropout) for _ in range(self.n_layers)]
        self.output_layer = layers.Dense(self.n_outputs, name="output")

    def call(self, inputs, training=None, mask=None):
        x = self.reshape_layer(inputs)  # (B, T, C)

        if self.model == 'itransformer':
            x = tf.transpose(x, [0, 2, 1])  # (B, C, T)
            means = tf.reduce_mean(x, axis=-1, keepdims=True)
            stdev = tf.sqrt(tf.reduce_mean(tf.square(x - means), axis=-1, keepdims=True) + 1e-5)
            x = (x - means) / stdev
            x = self._series_embed(x, training=training)
            x = self._channel_pe(x)
            x = tf.transpose(x, [0, 2, 1])  # back to (B, T, C)

        for hidden_l, dropout_l in zip(self.hidden_layers, self.dropout_hidden_layers):
            x = hidden_l(x, training=training)
            x = dropout_l(x, training=training)

        if getattr(self, "temporal_pool", None) is not None:
            x = self.temporal_pool(x)

        outputs = self.output_layer(x)

        if self.model == 'itransformer':
            n_targets = self.n_outputs // self.pred_len
            means_t = tf.gather(means, self.label_idxs, axis=1)
            stdev_t = tf.gather(stdev, self.label_idxs, axis=1)
            y = tf.reshape(outputs, [-1, self.pred_len, n_targets])
            means_b = tf.transpose(means_t, [0, 2, 1])
            stdev_b = tf.transpose(stdev_t, [0, 2, 1])
            y = y * stdev_b + means_b
            outputs = tf.reshape(y, [-1, self.n_outputs])

        return outputs
