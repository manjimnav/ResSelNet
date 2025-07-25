from glob import glob
import numpy as np
import torch
from lime import lime_tabular
import pickle
import shap

def lime_model(x):
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.ones((x.shape[0], trues.shape[1]))
    output =  model(x, y).detach().numpy()

    return output.reshape(output.shape[0], -1)

best_codes = [{"dataset_name": "airpollution", "code": "f0a7ccb66906bb5c23bd0a0e84c95a1a"}]

for row in best_codes:
    for folder_name in glob(f'results/{row["dataset_name"]}/best_results/{row["code"]}*'):
        print(folder_name)

        inputs = np.load(f"{folder_name}/inputs_test.npy")
        trues = np.load(f"{folder_name}/trues_test.npy")
        preds = np.load(f"{folder_name}/preds_test.npy")
        model = torch.load(open(f"{folder_name}/model.pth",'rb'))
        model.eval()
        feature_names_lagged = np.load(f"{folder_name}/feature_names.npy")
        target_names_lagged = np.load(f"{folder_name}/target_names.npy")

        target_names = {t.split(" ")[0][7:] for t in target_names_lagged}

        x_train = shap.kmeans(inputs.reshape(inputs.shape[0], -1), 5).data

        explainer = lime_tabular.LimeTabularExplainer(x_train, discretize_continuous=False, feature_names=feature_names_lagged.tolist(), class_names=target_names_lagged.tolist(), mode="classification")

        print(len(target_names_lagged))
        #explainer.save(open(f'{folder_name}/lime_explainer.pkl', 'wb'))

        lime_values =[]
        for i in range(x_train.shape[0]):
            explanations = explainer.explain_instance(x_train[i], lime_model, labels=list(range(len(target_names_lagged))), num_features=len(feature_names_lagged))

            lime_values_matrix = []
            for target, importances in explanations.as_map().items():
                importances = sorted(importances, key=lambda x: x[0])
                importances = np.array(list(map(lambda x: x[1], importances)))

                lime_values_matrix.append(importances)
            
            lime_values_matrix = np.stack(lime_values_matrix, axis=-1)
            lime_values.append(lime_values_matrix)

        lime_values = np.stack(lime_values)

        with open(f'{folder_name}/lime_values.pkl', 'wb') as f:
            pickle.dump(lime_values, f)