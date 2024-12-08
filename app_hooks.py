import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from utils.utils import get_file_names
from werkzeug.utils import secure_filename

os.chdir(os.path.dirname(__file__))

UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1000  # limits uploads to 50KB per file


# Landing page (endpoint /)
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# Predict Form
@app.route('/api/v1/predict_form/', methods=['GET'])
def get_prediction_form():
    return render_template('predictForm.html')


"""
Test request for /api/v1/predict:
http://127.0.0.1:5000/api/v1/predict?pressure=15&sun=60&mean_temp=80
"""


# Predict
@app.route('/api/v1/predict/', methods=['POST', 'GET'])
def make_prediction():
    if request.method == 'POST':
        # Form data
        pressure = request.form['pressure']
        sun = request.form['sun']
        mean_temp = request.form['mean_temp']

        # This is a test to see that it retrieves form info correctly, the prediction would go here instead

        # Redirection
        return redirect(url_for('make_prediction', pressure=pressure, sun=sun, mean_temp=mean_temp))

    # If method = GET, get data from the query parameters
    pressure = request.args.get('pressure', None)
    sun = request.args.get('sun', None)
    mean_temp = request.args.get('mean_temp', None)

    # Prepare the result as a dictionary
    result = {'pressure': pressure, 'sun': sun, 'mean_temp': mean_temp} if pressure and sun and mean_temp else None

    # Renders template with result
    return render_template('predict.html', result=result)


# Forecast
@app.route('/api/v1/forecast/', methods=['POST', 'GET'])
def forecast():
    forecast = None
    if request.method == 'POST':
        _dummy_1 = float(request.form['feature1'])
        _dummy_2 = float(request.form['feature2'])

        # prediction = model.predict(np.array([[feature1, feature2]]))

    return render_template('forecast.html', forecast=forecast)


# Update Data
@app.route('/api/v1/update_data/', methods=['POST', 'GET'])
def update_data():
    update_name = None
    if request.method == 'POST':
        f = request.files['newData']
        update_name = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], update_name))

    # List of uploaded files to select from
    data_op = get_file_names(UPLOAD_FOLDER)

    return render_template('updateData.html', update_name=update_name, data_op=data_op)


# Delete Data
@app.route('/api/v1/delete_data', methods=['POST'])
def delete_data():
    # List of uploaded files to select from
    data_op = get_file_names(UPLOAD_FOLDER)

    if request.method == 'POST':
        # List of files to delete
        to_delete = request.form.getlist('dataset_name')

        for file in to_delete:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    return jsonify({'Error': f'could not delete file {file}: {str(e)}'}), 500
            else:
                return jsonify({'Error': 'specified file does not exist'}), 404

        return redirect(url_for('update_data'))

    return render_template('updateData.html', data_op=data_op, delete_name=to_delete)


# Retrain
@app.route('/api/v1/retrain/', methods=['POST', 'GET'])
def retrain_model():
    metrics = None
    if request.method == 'POST':
        dataset_name = str(request.form.getlist('dataset_name')[0])
        print(dataset_name)

    # List of uploaded files to select from
    data_op = get_file_names(UPLOAD_FOLDER)

    return render_template('retrain.html', metrics=metrics, data_op=data_op)


# Webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    # Path to the repository where git pull will be performed
    path_repo = '/home/mardelmir1/Team_Challenge_Deployment'
    web_server = '/var/www/mardelmir1_pythonanywhere_com_wsgi.py'

    # Checks if the POST request contains JSON data
    if request.is_json:
        payload = request.json
        # Checks if the payload contains information about the repository
        if 'repository' in payload:
            # Extracts the repository name and cloning URL
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']

            # Changes to the repository directory
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'Repository directory does not exist!'}), 404

            # Does git pull on the repository
            try:
                subprocess.run(['git', 'pull', clone_url], check=True)
                subprocess.run(
                    ['touch', web_server], check=True
                )  # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'A git pull was performed on repository {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error while performing git pull on repository {repo_name}'}), 500
        else:
            return jsonify({'message': 'No repository information found in the payload'}), 400
    else:
        return jsonify({'message': 'The request does not contain JSON data'}), 400


if __name__ == '__main__':
    app.run()
