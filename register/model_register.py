import argparse, mlflow
mlflow.start_run()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    a = p.parse_args()

    model = mlflow.sklearn.load_model(a.model)
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="used_cars_price_prediction_model",
        artifact_path="random_forest_price_regressor"
    )

    print("Model registered.")
    mlflow.end_run()

if __name__ == "__main__":
    main()
