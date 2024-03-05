from tensorflow import keras
from .tf_models import BaseModel, TSLNet, ResSelNet

def get_tf_model(parameters: dict, n_features_in, n_features_out) -> keras.Model:
    
    
    selection_name = parameters['selection']['name']
    residual = parameters['selection'].get('params', dict()) or dict()
    residual = residual.get('residual', False)

    if selection_name == 'NoSelection':
        model = BaseModel(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    elif 'TimeSelectionLayer' in selection_name and not residual:
        model = TSLNet(parameters, n_features_in=n_features_in, n_features_out=n_features_out)
    elif 'TimeSelectionLayer' in selection_name and residual:
        model = ResSelNet(parameters, n_features_in=n_features_in, n_features_out=n_features_out)

    lr = parameters['model']['params']['lr']
    loss = keras.losses.MSE
    metrics = [keras.metrics.MSE, keras.metrics.MAE,
               keras.metrics.mean_absolute_percentage_error]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )

    return model