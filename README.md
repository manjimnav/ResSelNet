# A novel interpretable ozone forecasting approach based on deeplearning with masked residual connections

This repository contains all the code needed to reproduce the experiments in the paper presented by M. J. Jiménez-Navarro et al. 

> Air pollution is a growing threat, especially in low- and middle-income countries, causing 4.2 million premature deaths annually. Ground-level ozone is a major concern, necessitating accurate and interpretable prediction systems for public health. This study presents a machine learning-based forecasting approach applied in Andalusia, Spain, to improve ozone pollution predictions. Traditional time-series forecasting struggles with handling both linear and non-linear relationships in multi-horizon scenarios. Deep learning models often fail to adapt to these complexities. To address this, the proposed methodology integrates residual connections and feature selection, allowing features with linear relationships to bypass non-linear transformations while creating a hierarchical structure for better adaptability. This approach enhances flexibility, interpretability, and robustness, improving pollution level estimates. Additionally, it provides insights into the impact of individual features, supporting better decision-making in public health. The results demonstrate significant improvements in prediction accuracy and feature relevance assessment.

## Prerequisites

In order to run the experimentation several dependencies must be installed. The requirements has been listed in the `requirements.txt` which can be installed as the following:

```
pip install -r requirements.txt
```

The data should be uncompressed in a folder called "data" on the root directory of the project. The zip can be found at: [PollutionDataset](https://uses0-my.sharepoint.com/:u:/g/personal/mjimenez3_us_es/EVhnCJj0NHlNgFAqe5ncZTgBybUzicDoIS871Pc7IMHU4Q?e=xEWgab)

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

