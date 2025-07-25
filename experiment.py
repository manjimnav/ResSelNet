from src.experiment_launcher import ExperimentLauncher

if __name__ == '__main__':    
    config_path = 'configuration/dacnet/'
    experiment_launcher = ExperimentLauncher(config_path, save_file='results/airpollution/results_test.csv', search_type='bayesian', iterations=10)
    experiment_launcher.run()