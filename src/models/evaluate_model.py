import json
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def main():

    print("Starting model evaluation...")

    # Get the project root folder.
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    # Define folders.
    processed_folder = os.path.join(project_root, "data", "processed")
    models_folder = os.path.join(project_root, "models")
    metrics_folder = os.path.join(project_root, "metrics")

    # Define input files.
    model_file = os.path.join(models_folder, "model.pkl")
    x_test_file = os.path.join(processed_folder, "X_test_scaled.csv")
    y_test_file = os.path.join(processed_folder, "y_test.csv")

    # Define output files.
    predictions_file = os.path.join(processed_folder, "predictions.csv")
    scores_file = os.path.join(metrics_folder, "scores.json")

    # Check that the required files exist.
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Could not find the trained model: {model_file}"
        )

    if not os.path.exists(x_test_file):
        raise FileNotFoundError(
            f"Could not find the test features: {x_test_file}"
        )

    if not os.path.exists(y_test_file):
        raise FileNotFoundError(
            f"Could not find the test target: {y_test_file}"
        )

    # Create the metrics folder if necessary.
    os.makedirs(metrics_folder, exist_ok=True)

    # Load the model and test data.
    print("Loading model and test data...")

    model = joblib.load(model_file)
    X_test = pd.read_csv(x_test_file)
    y_test = pd.read_csv(y_test_file)

    # Convert y_test from a DataFrame to a Series.
    y_test = y_test.iloc[:, 0]

    # Check that both datasets contain the same number of rows.
    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test_scaled.csv and y_test.csv do not have "
            "the same number of rows."
        )

    # Make predictions.
    print("Making predictions...")
    predictions = model.predict(X_test)

    # Save the predictions.
    predictions_dataframe = pd.DataFrame(
        {"prediction": predictions}
    )

    predictions_dataframe.to_csv(
        predictions_file,
        index=False,
    )

    # Calculate the evaluation metrics.
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Store the metrics in a dictionary.
    scores = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    # Save the metrics as JSON.
    with open(scores_file, "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=4)

    print("Model evaluation completed successfully.")
    print(f"Predictions saved to: {predictions_file}")
    print(f"Scores saved to: {scores_file}")
    print(scores)


if __name__ == "__main__":
    main()
