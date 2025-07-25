from tensorflow.keras import layers
from .layer import get_time_selection_layer
from .base import BaseModel


class ResSelNet(BaseModel):
    def __init__(self, parameters, n_features_in=7, n_features_out=1):
        super().__init__(parameters, n_features_in, n_features_out)

        self.input_time_selection_layer = get_time_selection_layer(self.parameters, self.n_outputs, name=f'input_tsl')

        self.residual_hidden_layers = [get_time_selection_layer(self.parameters, self.n_outputs, name=f'tsl_{i}', flatten=parameters['model']['name'] == 'dense' or i==(self.n_layers-1)) for i in range(self.n_layers)]

    def call(self, inputs):

        inputs = self.reshape_layer(inputs)

        x = self.input_time_selection_layer(inputs) 

        for hidden_l, residual_l, dropout_l in zip(self.hidden_layers, self.residual_hidden_layers, self.dropout_hidden_layers):

            x = hidden_l(x)
            x = dropout_l(x)
            x = layers.Concatenate()([x, residual_l(inputs)])
        
        outputs = self.output_layer(x)

        return outputs
