import argparse
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from moe.predictor import Predictor
def load_pred(ckpt_path):
    c= torch.load(ckpt_path, map_location=torch.device("cuda"))
    print(f"Loading model: {c['model_hparams']}")
    predictor =  Predictor(
        model_name= c['model_name'],
        hparams= c['model_hparams'],
        optimizer_name= c['optimizer_name'],
        base_learning_rate= 1e-3,
        model_state= c['model_state_dict'],
        optimizer_state= c['optimizer_state_dict'],
        device='cuda',
        seed=0,
        debug=False
    )
    predictor.model.eval()
    return predictor
def evaluate_predictor(predictor, test_data_path, nnode):
    # 加载 test data（保持和前面的格式一致）
    with open(f"{test_data_path}{nnode}.pkl", "rb") as f:
        x_test, y_test = pickle.load(f)
    x_test = x_test.numpy()
    y_test = y_test.numpy()
    print(f"Loaded test data: x_test {x_test.shape}, y_test {y_test.shape}")

    # 推理
    x_tensor = torch.tensor(x_test).to(predictor.device).float()
    print("?")
    with torch.no_grad():
        y_pred = predictor.inference(x_tensor).cpu().numpy().squeeze()
    print(f"average label target value: {np.average(y_test)}")
    print(f"std of label target: {np.std(y_test)}")
    print(f"average predicted value: {np.average(y_pred)}")
    print(f"std of predicted : {np.std(y_pred)}")
    print(
        f"""Validation error:
        {predictor.model_name}:
        MAE = {mean_absolute_error(y_test, y_pred):.3f}
        MSE = {mean_squared_error(y_test, y_pred):.3f}
        """
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("-nnode", type=int, default=4)
    args = parser.parse_args()
    print(f"evaluating node {args.nnode}")
    # load model
    predictor = load_pred(args.model_file)
    evaluate_predictor(predictor,args.test_data,args.nnode)

if __name__ == '__main__':
    main()
