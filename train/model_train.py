import mlflow, argparse, os, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.start_run()

def first_file(path):
    return os.path.join(path, os.listdir(path)[0]) if os.path.isdir(path) else path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--test_data",  type=str, required=True)
    p.add_argument("--model_output", type=str, required=True)
    p.add_argument("--n_estimators", type=int, default=50)
    p.add_argument("--max_depth",   type=int, default=5)
    a = p.parse_args()

    train_df = pd.read_csv(first_file(a.train_data))
    test_df  = pd.read_csv(first_file(a.test_data))
    y_train, X_train = train_df["price"], train_df.drop(columns=["price"])
    y_test,  X_test  = test_df["price"],  test_df.drop(columns=["price"])

    model = RandomForestRegressor(
        n_estimators=a.n_estimators, max_depth=a.max_depth, random_state=42
    ).fit(X_train, y_train)

    mlflow.log_param("n_estimators", a.n_estimators)
    mlflow.log_param("max_depth",   a.max_depth)

    mse = mean_squared_error(y_test, model.predict(X_test))
    mlflow.log_metric("MSE", float(mse))
    print(f"MSE: {mse:.2f}")

    mlflow.sklearn.save_model(model, path=a.model_output)
    mlflow.end_run()

if __name__ == "__main__":
    main()
