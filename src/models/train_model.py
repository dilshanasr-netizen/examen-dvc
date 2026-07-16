import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main():


    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    processed_folder = os.path.join(
        project_root,
        "data",
        "processed",
    )

    models_folder = os.path.join(
        project_root,
        "models",
    )

    x_train_file = os.path.join(
        processed_folder,
        "X_train_scaled.csv",
    )

    y_train_file = os.path.join(
        processed_folder,
        "y_train.csv",
    )

    best_params_file = os.path.join(
        models_folder,
        "best_params.pkl",
    )

    model_file = os.path.join(
        models_folder,
        "model.pkl",
    )

    if not os.path.exists(x_train_file):
        raise FileNotFoundError(
            "Could not find the scaled training data:\n"
            f"{x_train_file}\n"
            "Run normalize_data.py before running train_model.py."
        )

    if not os.path.exists(y_train_file):
        raise FileNotFoundError(
            "Could not find the training target data:\n"
            f"{y_train_file}\n"
            "Run split_data.py before running train_model.py."
        )

    if not os.path.exists(best_params_file):
        raise FileNotFoundError(
            "Could not find the best parameters file:\n"
            f"{best_params_file}\n"
            "Run grid_search.py before running train_model.py."
        )

    os.makedirs(
        models_folder,
        exist_ok=True,
    )

    print("Loading scaled training data")
    X_train = pd.read_csv(x_train_file)

    print("Loading training target values")
    y_train = pd.read_csv(y_train_file)

    if X_train.empty:
        raise ValueError(
            "X_train_scaled.csv is empty."
        )

    if y_train.empty:
        raise ValueError(
            "y_train.csv is empty."
        )

    y_train = y_train.iloc[:, 0]

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train_scaled.csv and y_train.csv "
            "do not contain the same number of rows."
        )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print("Loading best parameters from grid search")
    best_parameters = joblib.load(best_params_file)


    print("Best parameters:")
    print(best_parameters)

    final_model = RandomForestRegressor(
        **best_parameters,
        random_state=42,
    )

    print("Training the model")
    final_model.fit(
        X_train,
        y_train,
    )

    print("Saving the model")
    joblib.dump(
        final_model,
        model_file,
    )

    print("Model training done")
    print(f"Trained model saved to: {model_file}")


if __name__ == "__main__":
    main()

