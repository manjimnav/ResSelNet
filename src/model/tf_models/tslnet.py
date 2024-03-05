import tensorflow as tf
from tensorflow.keras import layers
from .layer import get_time_selection_layer
from .base import BaseModel

class TSLNet(BaseModel):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.input_time_selection_layer = get_time_selection_layer(self.parameters, self.n_outputs, name=f'input_tsl')
        

    def call(self, inputs):

        inputs = self.reshape_layer(inputs)
        x = self.input_time_selection_layer(inputs) 

        for hidden_l, dropout_l in zip(self.hidden_layers, self.dropout_hidden_layers):

            x = hidden_l(x)
            x = dropout_l(x)
        
        outputs = self.output_layer(x)

        return outputs