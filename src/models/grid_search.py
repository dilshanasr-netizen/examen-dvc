import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main():

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    processed_folder = os.path.join(project_root, "data", "processed")
    models_folder = os.path.join(project_root, "models")

    x_train_file = os.path.join(processed_folder, "X_train_scaled.csv")
    y_train_file = os.path.join(processed_folder, "y_train.csv")

    best_params_file = os.path.join(models_folder, "best_params.pkl")

    if not os.path.exists(x_train_file):
        raise FileNotFoundError(
            f"Could not find the training file:\n{x_train_file}"
        )

    if not os.path.exists(y_train_file):
        raise FileNotFoundError(
            f"Could not find the target file:\n{y_train_file}"
        )

    os.makedirs(models_folder, exist_ok=True)

    print("Loading training data")
    X_train = pd.read_csv(x_train_file)
    y_train = pd.read_csv(y_train_file)

    y_train = y_train.iloc[:, 0]

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    model = RandomForestRegressor(random_state=42)

    parameter_grid = {
        "n_estimators": [100, 15-],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    # Create the Grid Search.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    print("Grid Search")
    grid_search.fit(X_train, y_train)

    best_parameters = grid_search.best_params_

    print("Best parameters found")
    print(best_parameters)

    joblib.dump(best_parameters, best_params_file)

    print("Grid Search done")
    print(f"Best parameters saved to: {best_params_file}")


if __name__ == "__main__":
    main()
