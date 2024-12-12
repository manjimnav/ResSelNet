import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import pi

sns.set()

pd.set_option('display.max_columns', 500)

def read_results_file(path:str="results/airpollution/results_test.csv") -> pd.DataFrame:

    total_metrics = pd.read_csv(path)

    total_metrics['model_name'] = total_metrics['model_name'].replace({'decisiontree': 'DT', 'lstm': "LSTM", 'dense': 'FF', 'lasso': 'L1'})
    total_metrics['selection_name'] = total_metrics['selection_name'].replace({'NoSelection': 'NS', 'TimeSelectionLayer': 'TSL', 'TimeSelectionLayerResidual': 'TSL'})

    if 'selection_params_residual' in total_metrics.columns:
        total_metrics.loc[(total_metrics.selection_name=='TSL') & (total_metrics.model_name=='LSTM')  & (total_metrics.selection_params_residual), 'model_name'] = 'LSTM*'
        total_metrics.loc[(total_metrics.selection_name=='TSL') & (total_metrics.model_name=='LSTM')& (~total_metrics.selection_params_residual.fillna(False)), 'model_name'] = 'TLSTM'
        total_metrics.loc[(total_metrics.selection_name=='TSL') & (total_metrics.model_name=='dense')& (total_metrics.selection_params_residual), 'model_name'] = 'FF*'
        total_metrics.loc[(total_metrics.selection_name=='TSL') & (total_metrics.model_name=='dense')& (~total_metrics.selection_params_residual.fillna(False)), 'model_name'] = 'TFF'
    
    total_metrics["n_features"] = total_metrics.selected_features.apply(eval).apply(count_features)

    return total_metrics

def get_best_results(metrics):

    groupby_cols = ["code"]

    agg_functions = {'dataset_name': lambda x: x.iloc[0], 
                            'selection_name': lambda x: x.iloc[0], 
                            'model_name': lambda x: x.iloc[0], 
                            'rmse_valid': 'mean'}
    selected_columns = ['dataset_name', 'selection_name', 'model_name', 'rmse', 'mae', 'mse', 'mape', 'wape', 'mase', 'dataset_params_test_year', 'dataset_params_seq_len', 'dataset_params_crossval', "dataset_params_shift","dataset_params_pred_len",  'selected_features', 'duration', 'n_features']
    grouped = metrics.groupby(groupby_cols, dropna=False)[['dataset_name', 'selection_name', 'model_name', "rmse_valid"]].agg(agg_functions)

    total_metrics_indexed = metrics.set_index(groupby_cols)

    best_results = total_metrics_indexed.loc[grouped.groupby(['dataset_name', 'selection_name', 'model_name'], dropna=False).rmse_valid.idxmin(), selected_columns].reset_index()

    return best_results

def count_features(feats):

    result = 0
    if type(feats) == list:
        result = len(feats)
    else:
        values = []
        for _,v in feats.items():
            values.extend(list(v))
        result = len(set(values))

    return result


def plot_experiments(total_metrics: pd.DataFrame, plot_type="cat", x='dataset_name', y="RMSE", col='dataset_name', hue='Model', kind="box", yscale='linear', xscale='linear', rename_map={'model_name': 'Model', 'rmse': 'RMSE', 'n_features': '# Features', 'duration': 'Duration'}):
    sns.set(font_scale=1.6, style='white')
    plt.figure(figsize=(20,20))

    total_metrics_renamed = total_metrics.rename(rename_map, axis=1)
    if plot_type == 'cat':
        g = sns.catplot(data=total_metrics_renamed, kind=kind, col=col, x=x, y=y, hue=hue, sharex=False, sharey=False, legend=True)
    else:
        g = sns.relplot(data=total_metrics_renamed, kind=kind, col=col, x=x, y=y, hue=hue, legend=True)

    g.set(xlabel=None, yscale=yscale, xscale=xscale)
    g.set_titles(template='')

    #sns.move_legend(g, "upper left", bbox_to_anchor=(.70, .45))

    plt.show()


def create_radar_chart(df, name_col, categories):
    # Ensure the name_col is in the DataFrame
    if name_col not in df.columns:
        raise ValueError(f"'{name_col}' column not found in the DataFrame")

    # Extract categories and number of metrics
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Set direction and offset
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set the category labels
    plt.xticks(angles[:-1], categories, size=10)

    # Plot each row in the DataFrame
    for row in df.index:
        values = df.loc[row, categories].values.flatten().tolist()
        max_values = df[categories].max().values
        normalized_values = (values / max_values).tolist()
        normalized_values += normalized_values[:1]  # Close the radar chart

        ax.plot(angles, normalized_values, 'o-', linewidth=2, label=df.loc[row, name_col])
        ax.fill(angles, normalized_values, alpha=0.2)

    # Define the scale for y-ticks
    max_value = 1  # since we're normalizing between 0 and 1
    ticks = np.linspace(0, max_value, 5)
    plt.yticks(ticks, [f"{tick:.2f}" for tick in ticks], color="grey", size=10)
    plt.ylim(0, max_value)

    # Add custom y-ticks for each category
    for i in range(N):
        angle = angles[i]
        max_val = df[categories[i]].max()
        tick_values = np.linspace(0, max_val, 5)
        for tick in tick_values:
            ax.text(angle, tick/max_val, f'{tick:.1f}', horizontalalignment='center', size=10, color='grey')

    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=False, shadow=False, ncol=1, fontsize=10, frameon=False)

    # Show the plot
    plt.show()
