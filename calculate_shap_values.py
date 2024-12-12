from glob import glob
import numpy as np
import torch
import shap
import pickle

def shap_model(x):

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.ones((x.shape[0], trues.shape[1]))
    return model(x, y).detach().numpy().reshape(x.shape[0], -1)

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

        print(inputs.shape)
        x_train = shap.kmeans(inputs.reshape(inputs.shape[0], -1), 20).data

        print(x_train.shape)
        explainer = shap.Explainer(shap_model, x_train)

        explainer.save(open(f'{folder_name}/explainer.pkl', 'wb'))

        shap_values = explainer(x_train, max_evals=x_train.shape[1]*2+1)

        with open(f'{folder_name}/shap_values.pkl', 'wb') as f:
            pickle.dump(shap_values, f)