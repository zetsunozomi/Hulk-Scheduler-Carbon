import argparse
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("-nnode", type=int, default=4)
    args = parser.parse_args()

    # load test data
    with open(f"{args.test_data}{args.nnode}.pkl", "rb") as f:
        x_test, y_test = pickle.load(f)
    print(f"Test data shape before reshape: {x_test.shape}")
    x_test = x_test.numpy().reshape(len(y_test), -1)
    y_test = y_test.numpy()
    print(f"Test data shape: x_test {x_test.shape}, y_test {y_test.shape}")

    # load model
    with open(f"{args.model_file}_{args.nnode}", "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model: {model.__class__.__name__}")

    # inference
    print("Prediction start...")
    y_pred = model.predict(x_test)
    print(f"average label target value: {np.average(y_test)}")
    print(f"std of label target: {np.std(y_test)}")
    print(f"average predicted value: {np.average(y_pred)}")
    print(f"std of predicted : {np.std(y_pred)}")
    print(
        f"""Validation error:
        {model.__class__.__name__}:
        MAE = {mean_absolute_error(y_test, y_pred):.3f}
        MSE = {mean_squared_error(y_test, y_pred):.3f}
        """
    )

if __name__ == '__main__':
    main()
