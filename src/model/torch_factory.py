import torch
from .torch_models import FullyConnected, LSTM, CNN, Attn, DACNetHierarchical, iTransformer
import lightning as L

class LightningWrapper(L.LightningModule):
    def __init__(self, model, params):
        super().__init__()

        self.model = model
        self.lr = params['model']['params']['lr']
        self.save_hyperparameters(ignore=['model'])

    def forward(self, inputs, target):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)

        loss = torch.nn.functional.mse_loss(output.view(-1), target.view(-1))

        if hasattr(self.model, 'custom_loss'):
            loss += self.model.custom_loss

        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.mse_loss(output.view(-1), target.view(-1))
        self.log("val/loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

def get_torch_model(parameters: dict, n_features_in, n_features_out) -> torch.nn.Module:

    if parameters['model']['name'] == "dense":          
        model = FullyConnected(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    elif parameters['model']['name'] == "lstm":          
        model = LSTM(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    elif "dacnet" in parameters['model']['name']:          
        model = DACNetHierarchical(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    elif "attn" in parameters['model']['name']:          
        model = Attn(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    elif "cnn" in parameters['model']['name']:          
        model = CNN(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    elif "itransformer" in parameters['model']['name']:          
        model = iTransformer(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    else:
        raise NotImplementedError()

    return LightningWrapper(model, parameters)
