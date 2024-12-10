# =====================================================================================================================================================================
# Imports
import csv
import json
import os
import pickle as pkl
from time import time

import numpy as np
from flask import jsonify
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

# =====================================================================================================================================================================
# Handle Data


def process_data(header, data_rows):
    """
    Process raw CSV files into numeric features(X) and target(y) arrays.

    Steps:
    - Identifies precipitation column index
    - Filters out date and snow_depth columns
    - Removes rows with missing values
    - Converts all remaining values to floats
    - Separates precipitation (y) from the other features(X)

    Returns:
    X (np.ndarray): Features matrix
    y (np.ndarray): Target vector (in this case, precipitation)
    """

    # Get precipitation original index in header
    precip_col_idx = header.index('precipitation')

    # Get the column indices for the desired features in the dataset
    feature_idxs = [i for i, feat_name in enumerate(header) if feat_name not in ['date', 'snow_depth']]

    # Get precipitation index in filtered feature set
    precip_idx = feature_idxs.index(precip_col_idx)

    # Define missings/nan value markers
    missings = [None, '', 'NaN', 'nan', 'N/A', 'n/a', 'NULL', 'null']

    # Initialise list to add value lists (rows) to
    all_values = []

    # Iterate through each data row
    for row in data_rows:
        # Extract only desired columns
        values = [row[i] for i in feature_idxs]
        # Skip any rows with missings
        if any(val in missings for val in values):
            continue
        # Convert al values to floats
        values = [float(val) for val in values]
        # Add cleaned row to al_values
        all_values.append(values)

    # Initialise X, y lists
    X = []
    y = []

    # Iterate over rows and remove precipitation value from features
    for row in all_values:
        feature_vals = [val for i, val in enumerate(row) if i != precip_idx]
        target_val = row[precip_idx]
        X.append(feature_vals)
        y.append(target_val)

    # Convert lists to arrays for modeling
    X = np.array(X)
    y = np.array(y)

    return X, y


def get_file_names(upload_folder) -> list:
    data_op = [file for file in os.listdir(upload_folder) if file.endswith('.csv')]
    return data_op if len(data_op) != 0 else None


# =====================================================================================================================================================================
# Cross-Validate


def cross_validate_models(models: dict, scaler, X_train, y_train) -> dict:
    """
    Performs cross_validation on a dictionary of algorithms {name: algorithm}
    Uses the scaler used in initial training of first model
    Uses predefined metrics for cross-validation

    Steps:
    - For each algorithm in the dictionary, creates a pipeline with the scaler and the algorithm
    - Performs cross-validation and records MSE, RMSE and MAPE)

    Returns:
    cv_results (dict): Dictionary of model names and their mean error metrics)
    pipelines (dict): Dictionary of each models name and pipeline.
    """

    model_names = []
    mse = []
    rmse = []
    mape = []
    pipelines = {}

    for name, alg in models.items():
        # Create a pipeline for each algorithm
        pipe = Pipeline(steps=[('scaler', scaler), ('regressor', alg)])

        # Cross-validates using MSE, RMSE and MAPE as metrics
        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            scoring=('neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error'),
        )

        # Update all the lists
        model_names.append(name)
        mse.append(-np.mean(cv_scores['test_neg_mean_squared_error']))
        rmse.append(-np.mean(cv_scores['test_neg_root_mean_squared_error']))
        mape.append(-np.mean(cv_scores['test_neg_mean_absolute_percentage_error']))
        pipelines[name] = pipe

    # Organise lists as a dictionary to resemble JSON data structure
    cv_results = {'Model': model_names, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    return cv_results, pipelines


# =====================================================================================================================================================================
# Tune Hyperparams


def tune_hyperparameters(pipelines: dict, param_grids: dict, X_train, y_train, cv_scoring: str) -> dict:
    """
    Performs hyperparameter tuning on pipelines using GridSearchCV.
    Allows user defined coring metric

    Steps:
    - For each pipeline, checks if there's a parameter grid
    - Run a grid search to find the best hyperparameters
    - If no grid is found, just fits the pipeline directly
    - Prints timing and best parameters for each model

    Returns:
    tuned_models (dict): Dictionary of model names and tuned models
    """
    tuned_models = {}

    for name, pipe in pipelines.items():
        params = param_grids.get(name)
        if params:
            print(f'Tuning {name} hyperparameters...')
            gs = GridSearchCV(pipe, param_grid=params, cv=5, scoring=cv_scoring)

            start = time()
            gs.fit(X_train, y_train)
            end = time()

            tuning_time = end - start
            time_message = (
                f'Tuning {name} took: {tuning_time:.3f} seconds'
                if tuning_time < 60
                else f'Tuning {name} took: {tuning_time/60:.3f} minutes'
            )

            best = gs.best_estimator_

            print('----Hyperparameter tuning complete ----')
            print(time_message)
            if 'neg_' in cv_scoring:
                print(f'Best Score: {-gs.best_score_:.5f}')
            else:
                print(f'Best Score: {gs.best_score_:.5f}')
            print(f'Best parameters:\n{gs.best_params_}')
            print()

        else:
            print(f'No parameter grid found for {name}. Fitting model directly...')

            start = time()
            cv = cross_validate(pipe, X_train, y_train, scoring=cv_scoring)
            pipe.fit(X_train, y_train)
            best = pipe
            end = time()

            tuning_time = end - start
            time_message = (
                f'Fitting {name} took: {tuning_time:.3f} seconds'
                if tuning_time < 60
                else f'Fitting {name} took: {tuning_time/60:.3f} minutes'
            )
            print(time_message)

            # Prints best scores from cross-validation
            if 'neg_' in cv_scoring:
                print(f"Best Score: {-np.mean(cv['test_score']):.5f}")
            else:
                print(f"Best Score: {np.mean(cv['test_score']):.5f}")
            print()

        tuned_models[name] = best

    return tuned_models


# =====================================================================================================================================================================
# Evaluate on test set


def test_evaluation(tuned_models: dict, X_train=None, y_train=None, X_test=None, y_test=None) -> dict:
    """
    Fits the tuned models on training data and evaluates them on the test set.

    Steps:
    - If tuned_models is a single model (pipeline), convert it to a dict.
    - Fit each model on X_train, y_train.
    - Predict on X_test and compute MSE, RMSE, MAPE.

    Returns:
    - evaluation_results (dict): Dictionary of evaluation results for each model.
    """
    model_names = []
    mse = []
    rmse = []
    mape = []

    # Accounts for when i want to use this function and there is only one model
    # This will actually be the majority use case but i wanted to make a function that allows for evaluation of mutiple models
    if not isinstance(tuned_models, dict):
        tuned_models = {f'{tuned_models.steps[-1][1].__class__.__name__}': tuned_models}

    for name, model in tuned_models.items():
        # Fit on training data
        model.fit(X_train, y_train)
        # Predict on test data
        y_preds = model.predict(X_test)

        # Calculate evalution metrics
        mse_score = mean_squared_error(y_test, y_preds)
        rmse_score = root_mean_squared_error(y_test, y_preds)
        mape_score = mean_absolute_percentage_error(y_test, y_preds)

        model_names.append(name)
        mse.append(mse_score)
        rmse.append(rmse_score)
        mape.append(mape_score)

    # Organise into dictionary and return metrics
    evaluation_results = {'Model': model_names, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    return evaluation_results


# =====================================================================================================================================================================
# Save best model


def save_best_model(
    evaluation_results: dict, tuned_models: dict, selection_metric='', save_path='./models/best_model.pkl'
):
    """
    Selects the best model based on a specified metric and saves it

    Steps:
    - If tuned_models is a dictionary, finds the model with the best (lowest) metric score
    - If it's a single model, just uses that model
    - Save the selected model as a pickle file

    Returns:
    best_model (pipeline): Pipeline of best performing estimator
    """
    if not isinstance(tuned_models, dict):
        best_model = tuned_models
    else:
        # Finds the index of the best model based on the specified selection_metric
        min_idx = np.argmin(evaluation_results[selection_metric])
        best_model_name = evaluation_results['Model'][min_idx]
        best_model = tuned_models[best_model_name]

    # Sves the best model
    pkl.dump(best_model, open(save_path, 'wb'))

    return best_model


# =====================================================================================================================================================================
# Load evaluation_results


def load_evaluation_results():
    """
    Loads previously saved evaluation results from a JSON file.
    If the file doesn’t exist, returns None.
    """
    try:
        with open('./data/evaluation_results.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None


# =====================================================================================================================================================================
# Save evaluation_results


def save_evaluation_results(results):
    """
    Saves evaluation results (a dictionary) to a JSON file.
    """
    with open('./data/evaluation_results.json', 'w') as file:
        json.dump(results, file, indent=4)


# =====================================================================================================================================================================
# Retrain
def retrain_model(model, scaler, file_path=None):
    if file_path is not None:
        # Read uploaded CSV file
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            new_header = next(reader)  # Extract the header
            new_data = [row for row in reader]  # Extract the data
            print('Uploaded data read successfully')
    else:
        # Read default new CSV file
        with open('./data/new_data.csv', 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            new_header = next(reader)  # Extract the header
            new_data = [row for row in reader]  # Extract the data
            print('New default data read successfully')

    # Read the original CSV file
    with open('./data/original_data.csv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        original_header = next(reader)  # Extract the header
        original_data = [row for row in reader]  # Extract the data
        print('Original data read successfully')

    # Check that the headers match to ensure data can be merged
    if new_header != original_header:
        return jsonify({'Error': 'The new dataset is missing or has extra columns'}), 400

    # Merge original and new data
    updated_data = original_data + new_data
    print('Data merged successfully')

    # Process the data using utils function process_data
    X, y = process_data(original_header, updated_data)

    # Scale data with original scaler
    X = scaler.transform(X)

    # Split data into X, y pairs for train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    print('Merged data split successfully')

    evaluation_results = test_evaluation(model, X_train, y_train, X_test, y_test)
    new_model = model.fit(X_train, y_train)

    return evaluation_results, new_model
