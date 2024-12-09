# Team Challenge: Deployment

## Overview

This repository focuses on deploying a machine learning model on PythonAnywhere, serving the model as a web application.


## Endpoints

- `/`: landing page, explains all possible endpoints.
- `/predict`: makes a prediction based on specific arguments given by the user.
- `/retrain`: retrains model with a mix of the original data and new data.
- `/verify_model`: to choose which model to keep, the original model or the retrained one. 

#### Table test 1

| Endpoint               | Explanation                        |
| ---------------------- | ---------------------------------- |
| `/`                    | Landing page                       |
| `/api/v1/predict_form` | Collects input to make predictions |
| `/api/v1/predict`      | Makes predictions                  |
| `/api/v1/update_data`  | Uploads new data (csv files)       |
| `/api/v1/retrain`      | Retrains model                     |
| `/webhook`             | Automates updates                  |


#### Table test 2
<table class="endpoint-table">
    <tr>
        <td class="endpoint">/</td>
        <td>Landing page</td>
    </tr>
    <tr>
        <td class="endpoint">/api/v1/predict_form</td>
        <td>Endpoint for collecting input to make predictions</td>
    </tr>
    <tr>
        <td class="endpoint">/api/v1/predict</td>
        <td>Endpoint for making predictions</td>
    </tr>
    <tr>
        <td class="endpoint">/api/v1/forecast</td>
        <td>Endpoint for getting forecasts</td>
    </tr>
    <tr>
        <td class="endpoint">/api/v1/update_data</td>
        <td>Endpoint for uploading new data (csv files)</td>
    </tr>
    <tr>
        <td class="endpoint">/api/v1/retrain</td>
        <td>Endpoint for retraining the model</td>
    </tr>
    <tr>
        <td class="endpoint">/webhook</td>
        <td>Endpoint for automated updates</td>
    </tr>
</table>


### Authors
- [carlosguerrasoria](https://github.com/carlosguerrasoria)
- [mailliwJ](https://github.com/mailliwJ)
- [mardelmir](https://github.com/mardelmir)