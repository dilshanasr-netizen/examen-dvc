import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():

    print("Starting data normalization")

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    processed_folder = os.path.join(project_root, "data", "processed")
    models_folder = os.path.join(project_root, "models")

    # Define the input files.
    x_train_file = os.path.join(processed_folder, "X_train.csv")
    x_test_file = os.path.join(processed_folder, "X_test.csv")

    # Define the output files.
    x_train_scaled_file = os.path.join(processed_folder, "X_train_scaled.csv")
    x_test_scaled_file = os.path.join(processed_folder, "X_test_scaled.csv")
    scaler_file = os.path.join(models_folder, "scaler.pkl")

    if not os.path.exists(x_train_file):
        raise FileNotFoundError(
            f"Could not find the training file:\n{x_train_file}"
        )

    if not os.path.exists(x_test_file):
        raise FileNotFoundError(
            f"Could not find the test file:\n{x_test_file}"
        )

    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)

    print("Loading training data")
    X_train = pd.read_csv(x_train_file)

    print("Loading test data")
    X_test = pd.read_csv(x_test_file)

    if X_train.empty:
        raise ValueError("X_train.csv is empty.")

    if X_test.empty:
        raise ValueError("X_test.csv is empty.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train.csv and X_test.csv do not have the same columns."
        )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    scaler = StandardScaler()

    print("Fitting the scaler on the training data")
    X_train_scaled = scaler.fit_transform(X_train)

    print("Scaling the test data")
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
    )

    X_train_scaled.to_csv(
        x_train_scaled_file,
        index=False,
    )

    X_test_scaled.to_csv(
        x_test_scaled_file,
        index=False,
    )

    joblib.dump(
        scaler,
        scaler_file,
    )

    print("Data normalization done")
    print(f"Scaled training data saved to: {x_train_scaled_file}")
    print(f"Scaled test data saved to: {x_test_scaled_file}")
    print(f"Scaler saved to: {scaler_file}")


if __name__ == "__main__":
    main()
