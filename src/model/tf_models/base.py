import tensorflow as tf
from tensorflow.keras import layers
from .layer import get_base_layer

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

        # Layers instantiation
        self.n_outputs = n_features_out*self.pred_len
        self.reshape_layer = layers.Reshape((self.seq_len, n_features_in), name='inputs_reshaped') # We expect the input size to be seq_len*feat_dim. So we transform it to (seq_len, feat_dim)

        if self.model == 'dense':
            self.hidden_layers = [self.BASE_LAYER(self.n_units, name=f'layer_{i}') for i in range(self.n_layers)]
            self.reshape_layer = layers.Reshape((self.seq_len*n_features_in, ), name='inputs_reshaped') # We expect the input size to be seq_len*feat_dim. So we transform it to (seq_len, feat_dim)

        else:

            self.hidden_layers = [self.BASE_LAYER(self.n_units, name=f'layer_{i}', return_sequences=True if i<(self.n_layers-1) else False) for i in range(self.n_layers)]
            self.reshape_layer = layers.Reshape((self.seq_len, n_features_in), name='inputs_reshaped') # We expect the input size to be seq_len*feat_dim. So we transform it to (seq_len, feat_dim)

        
        self.dropout_hidden_layers = [layers.Dropout(self.dropout) for _ in range(self.n_layers)]

        self.output_layer = layers.Dense(self.n_outputs, name="output")


    def call(self, inputs):

        inputs = self.reshape_layer(inputs)

        x = inputs
        for hidden_l, dropout_l in zip(self.hidden_layers, self.dropout_hidden_layers):

            x = hidden_l(x)
            x = dropout_l(x)
        
        outputs = self.output_layer(x)

        return outputs