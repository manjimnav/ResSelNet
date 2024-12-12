import torch
from torch import nn

class flat_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor.flatten(start_dim=1)

class LSTM(nn.Module):
    def __init__(self, parameters: dict,  n_features_in: int, n_features_out: int):
        super().__init__()

        n_layers = parameters['model']['params']['layers']
        n_units = parameters['model']['params']['units']
        dropout = parameters['model']['params']['dropout']
        self.pred_len = parameters['dataset']['params']['pred_len']
        seq_len = parameters['dataset']['params']['seq_len']
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out

        module_list = [nn.LSTM(n_features_in, n_units, batch_first=True, num_layers=n_layers, dropout=dropout), flat_tensor()]

        module_list.append(nn.Flatten())
        module_list.append(nn.Linear(n_units*seq_len, self.n_features_out*self.pred_len))

        self.sequential = nn.Sequential(*module_list)

        self.double()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in x n_features
        # output of shape batch_size x n_timesteps_out x n_features out

        return self.sequential(x).reshape(x.shape[0], self.pred_len, self.n_features_out)