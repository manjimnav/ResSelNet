import torch
from torch import nn
from .layers import ConvBlock

class CNN(nn.Module):
    def __init__(self, parameters: dict,  n_features_in: int, n_features_out: int):
        super().__init__()

        n_layers = parameters['model']['params']['layers']
        n_units = parameters['model']['params']['units']
        dropout = parameters['model']['params']['dropout']
        self.pred_len = parameters['dataset']['params']['pred_len']
        seq_len = parameters['dataset']['params']['seq_len']
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out

        module_list = [ConvBlock(n_features_in if l==0 else n_units, n_units, dropout=dropout) for l in range(n_layers)]

        module_list.append(nn.Flatten())
        module_list.append(nn.Linear(n_units*seq_len, self.n_features_out*self.pred_len))

        self.sequential = nn.Sequential(*module_list)

        self.double()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in x n_features
        # output of shape batch_size x n_timesteps_out x n_features out

        return self.sequential(x).reshape(x.shape[0], self.pred_len, self.n_features_out)