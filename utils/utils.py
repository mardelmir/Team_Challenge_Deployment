import os
import pickle as pkl
from time import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

# =====================================================================================================================================================================
# Handle Data


def process_data(X: pd.DataFrame) -> pd.DataFrame:
    X = X.dropna(axis=0).copy()
    X['pressure'] = X['pressure'] / 1000
    return X


def load_zipped_data(zip_path: str) -> pd.DataFrame:
    with ZipFile(zip_path, 'r') as z:
        with z.open(z.namelist()[0]) as f:
            data = pd.read_csv(f)
    return data


def save_to_zip(data: pd.DataFrame, zip_path: str, file_name: str):
    data.to_csv(file_name, index=False)
    with ZipFile(zip_path, 'w') as z:
        z.write(file_name, arcname=file_name)
    os.remove(file_name)


# =====================================================================================================================================================================
# Cross-Validate

scaler = pkl.load(open('../transformers/scaler.pkl', 'rb'))


def cross_validate_models(models: dict, X_train, y_train) -> pd.DataFrame:
    model_names = []
    mse = []
    rmse = []
    mape = []
    pipes = {}

    for name, alg in models.items():
        pipe = Pipeline(steps=[('scaler', scaler), ('regressor', alg)])

        CVresults = cross_validate(
            pipe,
            X_train,
            y_train,
            scoring=('neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error'),
        )

        model_names.append(name)
        mse.append(-np.mean(CVresults['test_neg_mean_squared_error']))
        rmse.append(-np.mean(CVresults['test_neg_root_mean_squared_error']))
        mape.append(-np.mean(CVresults['test_neg_mean_absolute_percentage_error']))
        pipes[name] = pipe

    cvResultsDF = pd.DataFrame({'Model': model_names, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape})

    return cvResultsDF, pipes


# =====================================================================================================================================================================
# Tune Hyperparams


def tune_hyperparameters(pipelines: dict, param_grids: dict, X_train, y_train, cv_scoring: str) -> dict:
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

            if 'neg_' in cv_scoring:
                print(f'Best Score: {-np.mean(cv['test_score']):.5f}')
            else:
                print(f'Best Score: {np.mean(cv['test_score']):.5f}')
            print()

        tuned_models[name] = best

    return tuned_models


# =====================================================================================================================================================================
# Evaluate on test set


def test_evaluation(tuned_models: dict, X_train=None, y_train=None, X_test=None, y_test=None) -> pd.DataFrame:
    model_names = []
    mse = []
    rmse = []
    mape = []

    refitted_models = {}

    if not isinstance(tuned_models, dict):
        tuned_models = {f'{tuned_models.steps[-1][1].__class__.__name__}': tuned_models}

    for name, model in tuned_models.items():
        refitted_models[name] = model.fit(X_train, y_train)
        y_preds = refitted_models[name].predict(X_test)

        mse_score = mean_squared_error(y_test, y_preds)
        rmse_score = root_mean_squared_error(y_test, y_preds)
        mape_score = mean_absolute_percentage_error(y_test, y_preds)

        model_names.append(name)
        mse.append(mse_score)
        rmse.append(rmse_score)
        mape.append(mape_score)

    predResultsDF = pd.DataFrame({'Model': model_names, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape})

    return predResultsDF, refitted_models


# =====================================================================================================================================================================
# Save best model


def save_best_model(
    test_resultsDF: pd.DataFrame, tuned_models: dict, selection_metric='', save_path='../models/best_model.pkl'
):
    if not isinstance(tuned_models, dict):
        best_model = tuned_models
    else:
        best_model = tuned_models[test_resultsDF.loc[test_resultsDF[f'{selection_metric}'].idxmin(), 'Model']]
    pkl.dump(best_model, open(save_path, 'wb'))
