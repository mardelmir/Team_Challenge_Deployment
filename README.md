# Team Challenge: Deployment

## Overview

This repository focuses on deploying a machine learning model on PythonAnywhere, serving the model as a web application.


## Endpoints

| Endpoint               | Methods       | Explanation                               |
| ---------------------- | ------------- | ----------------------------------------- |
| `/`                    | `GET`         | Landing page                              |
| `/api/v1/predict_form` | `GET`, `POST` | Collects input to make predictions (form) |
| `/api/v1/predict`      | `GET`, `POST` | Makes predictions based on trained model  |
| `/api/v1/update_data`  | `GET`, `POST` | Uploads new data (csv files)              |
| `/api/v1/delete_data`  | `POST`        | Deletes previously uploaded files (form)  |
| `/api/v1/retrain`      | `GET`, `POST` | Retrains model                            |
| `/webhook`             | `POST`        | Automates updates                         |


### Authors
- [carlosguerrasoria](https://github.com/carlosguerrasoria)
- [mailliwJ](https://github.com/mailliwJ)
- [mardelmir](https://github.com/mardelmir)