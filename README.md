# A novel interpretable deep learning approach for time series forecasting with masked residual connections

This repository contains all the code needed to reproduce the experiments in the paper presented by M. J. Jiménez-Navarro et al. 

> The time series forecasting problem remains one of the most challenging issues in machine learning. Despite astonishing advances in areas like artificial vision and natural language processing due to the emergence of deep learning, there is still no silver bullet for time series forecasting. In these problems, the data may contain features that are linearly related to the target and others with non-linear relationships.
Deep learning often struggles to handle linearly related features and adapt to the peculiarities of the problem, especially in time series forecasting, where past values of the target are used as input. We propose a general-purpose methodology for deep learning models that combines residual connections and feature selection to enhance the model's flexibility. The residual connection allows features with highly linear relationships to bypass non-linear transformations while relevant features can be selected at different depths, creating a flexible hierarchy, which allows the model to detect the linear and non-linear relationships automatically. The results demonstrate a consistent improvement in most datasets, which makes this methodology more robust to the peculiarities of the problem. Additionally, the model provides in-depth information on the relevant features and the depth to which they were selected.

## Prerequisites

In order to run the experimentation several dependencies must be installed. The requirements has been listed in the `requirements.txt` which can be installed as the following:

```
pip install -r requirements.txt
```

## Reproduce result

#### Reproduce full experimentation

The experimentation is performed via the `experiment.ipynb` notebook which runs the methodology over all datasets and models. The experimentation is made via the `ExperimentLauncher` receiving the following parameters:

* ``config_path``: The path to the three configuration files for the experimentation corresponding to: `data_config.yaml`, `model_config.yaml` and `selection_config.yaml`.
    * ``data_config``: Enumerate all the datasets employed in during the experimentation and its hyperparameter ranges.
    * ``model_config``: Enumerate the models employed and its hyperparameter ranges.
    * ``selection_config``: Enumerate the selection methods and its hyperparameter ranges.

* `save_file`: The csv file which contains all the results obtained in each experimentation.

* `seach_type`: The type of seach over the hyperparameters ranges performed, can be one of bayesian or grid.

* `iterations`: The number of iterations to run the hyperparameter search (only used in bayesian).

#### Analysis

Finally, a notebook with the analysis performed in the paper is provided in the `analysis.ipynb` notebook.

