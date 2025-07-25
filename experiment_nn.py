from src.experiment_launcher import ExperimentLauncher

if __name__ == '__main__':    
    config_path = 'configuration/airpollution/'
    experiment_launcher = ExperimentLauncher(config_path, save_file='results/airpollution/results_nn_v2.csv', search_type='bayesian', iterations=25)
    experiment_launcher.run()