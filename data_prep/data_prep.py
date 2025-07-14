import os, argparse, logging, mlflow, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

mlflow.start_run()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--test_train_ratio", type=float, default=0.2)
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--test_data", type=str, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.data)

    if "Segment" in df.columns:
        df["Segment"] = LabelEncoder().fit_transform(df["Segment"])

    train_df, test_df = train_test_split(
        df, test_size=args.test_train_ratio, random_state=42)

    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data,  exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data,  "test.csv"),  index=False)

    mlflow.log_metric("train_rows", train_df.shape[0])
    mlflow.log_metric("test_rows",  test_df.shape[0])
    logging.info("Data prep complete.")
    mlflow.end_run()

if __name__ == "__main__":
    main()
