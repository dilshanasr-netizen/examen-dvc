import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main():


    # Get the root folder of the project.
    # This script is inside src/models/,
    # so we go up three folder levels.
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    # Define the main project folders.
    processed_folder = os.path.join(
        project_root,
        "data",
        "processed",
    )

    models_folder = os.path.join(
        project_root,
        "models",
    )

    # Define the input file paths.
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

    # Define the output file path.
    model_file = os.path.join(
        models_folder,
        "model.pkl",
    )

    # Check that the required input files exist.
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

    # Create the models folder if it does not already exist.
    os.makedirs(
        models_folder,
        exist_ok=True,
    )

    # Load the scaled training features.
    print("Loading scaled training features...")
    X_train = pd.read_csv(x_train_file)

    # Load the training target.
    print("Loading training target...")
    y_train = pd.read_csv(y_train_file)

    # Check that the datasets are not empty.
    if X_train.empty:
        raise ValueError(
            "X_train_scaled.csv is empty."
        )

    if y_train.empty:
        raise ValueError(
            "y_train.csv is empty."
        )

    # y_train was loaded as a DataFrame.
    # Select its first column to convert it into a Series.
    y_train = y_train.iloc[:, 0]

    # Check that the feature and target datasets
    # contain the same number of rows.
    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train_scaled.csv and y_train.csv "
            "do not contain the same number of rows."
        )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Load the parameters found during Grid Search.
    print("Loading best parameters...")
    best_parameters = joblib.load(best_params_file)

    # The saved parameters should be a dictionary.
    if not isinstance(best_parameters, dict):
        raise ValueError(
            "The contents of best_params.pkl are not a dictionary."
        )

    print("Best parameters:")
    print(best_parameters)

    # Create the final Random Forest model.
    # The ** operator passes the dictionary values
    # as model parameters.
    final_model = RandomForestRegressor(
        **best_parameters,
        random_state=42,
    )

    # Train the final model.
    print("Training the final model...")
    final_model.fit(
        X_train,
        y_train,
    )

    # Save the trained model.
    print("Saving the trained model...")
    joblib.dump(
        final_model,
        model_file,
    )

    print("Model training completed successfully.")
    print(f"Trained model saved to: {model_file}")


if __name__ == "__main__":
    main()

