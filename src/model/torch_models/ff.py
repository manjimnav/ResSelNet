import torch
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, parameters: dict,  n_features_in: int, n_features_out: int):
        super().__init__()

        n_layers = parameters['model']['params']['layers']
        n_units = parameters['model']['params']['units']
        dropout = parameters['model']['params']['dropout']
        self.pred_len = parameters['dataset']['params']['pred_len']
        self.seq_len = parameters['dataset']['params']['seq_len']
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out

        module_list = []

        layer_input_size = self.seq_len*n_features_in
        for _ in range(n_layers):
            module_list.extend([nn.Linear(layer_input_size, n_units), nn.ReLU(), nn.Dropout(dropout)])
            layer_input_size = n_units

        module_list.append(nn.Linear(n_units, self.n_features_out*self.pred_len))

        self.sequential = nn.Sequential(*module_list)

        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        x = x.reshape(x.shape[0], -1)
        return self.sequential(x).reshape(x.shape[0], self.pred_len, self.n_features_out)