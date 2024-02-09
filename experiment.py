from src.experiment import ExperimentLauncher
config_path = 'configuration/ResSelNet/'
experiment_launcher = ExperimentLauncher(config_path, save_file='results/ResSelNet/results_test.csv', search_type='bayesian', iterations=25)
experiment_launcher.run()