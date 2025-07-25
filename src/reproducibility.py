from glob import glob
import yaml
from src.experiment_instance import ExperimentInstance
import numpy as np
import torch
import random
import os
import pickle

def store_best_model_data(best_codes_by_model, seed=123):
    for row in best_codes_by_model:

        config_file = glob(f'configuration/{row["dataset_name"]}/**/{row["code"]}*', recursive=True)[0]
        print(config_file)
        with open(config_file) as stream:
            try:
                experiment_params = yaml.load(stream, Loader=yaml.Loader)

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                experiment = ExperimentInstance(experiment_params)

                print(f"Reproducing and saving: {row['code']}")

                _ = experiment.run()

                raw_results = experiment.raw_results_

                for test_year, dates, inputs, trues, preds in raw_results:
                    code = f'{row["code"]}-{test_year}'

                    folder_name = f'results/{row["dataset_name"]}/best_results/{code}'

                    if not os.path.isdir(folder_name):
                        os.mkdir(folder_name)

                    with open(f'{folder_name}/dates.npy', 'wb') as f:
                        np.save(f, dates)
                    with open(f'{folder_name}/inputs_test.npy', 'wb') as f:
                        np.save(f, inputs)
                    with open(f'{folder_name}/trues_test.npy', 'wb') as f:
                        np.save(f, trues)
                    with open(f'{folder_name}/preds_test.npy', 'wb') as f:
                        np.save(f, preds)
                    with open(f'{folder_name}/feature_names.npy', 'wb') as f:
                        np.save(f, experiment.dataset.feature_names)
                    with open(f'{folder_name}/target_names.npy', 'wb') as f:
                        np.save(f, experiment.dataset.target_names)
                    
                    if experiment.parameters['model']['params']['type'] == 'sklearn':
                        with open(f'{folder_name}/model.pth', 'wb') as f:
                            pickle.dump(experiment.model, f)
                    else:
                        torch.save(experiment.model, f'{folder_name}/model.pth')

            except yaml.YAMLError as exc:
                print(exc)