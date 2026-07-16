from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import os


def main():


#    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )))

    input_file = os.path.join( project_root, "data", "raw", "raw.csv", )
    output_folder = os.path.join( project_root, "data", "processed", )

    if not os.path.exists(input_file): 
        raise FileNotFoundError( f"Could not find the dataset at: {input_file}" )
    
    os.makedirs(output_folder, exist_ok=True)
    data = pd.read_csv(input_file)
    target_column = "silica_concentrate"

    if target_column not in data.columns: 
        raise ValueError( f"The target column '{target_column}' was not found." )

    # Separate the target column from the feature columns.
    X = data.drop(columns=[target_column])
    y = data[target_column]

    non_numeric_columns = X.select_dtypes(exclude=["number"]).columns

    if len(non_numeric_columns) > 0:
        print(
        "Removing non-numeric columns:",
        list(non_numeric_columns),
    )

    X = X.drop(columns=non_numeric_columns)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, )
    X_train.to_csv(
    os.path.join(output_folder, "X_train.csv"),
    index=False,
    )

    X_test.to_csv(
        os.path.join(output_folder, "X_test.csv"),
        index=False,
    )
    
    y_train.to_frame().to_csv(
        os.path.join(output_folder, "y_train.csv"),
        index=False,
    )
    
    y_test.to_frame().to_csv(
        os.path.join(output_folder, "y_test.csv"),
        index=False,
    ) 

    print("Data split completed successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")


if __name__ == "__main__":
    main()
