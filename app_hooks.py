import json
import os
import pickle
import subprocess

import numpy as np
import utils.utils_V2 as u
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

os.chdir(os.path.dirname(__file__))

SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
UPLOAD_FOLDER = './data/uploads'
TEMP_MODEL_PATH = './models/temp_model.pkl'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 350 * 1000  # limits uploads to 350KB per file


# Landing page (endpoint /)
@app.route('/', methods=['GET'])
def home():
    message = request.args.get('message')
    return render_template('index.html', message=message)


# Predict Form
@app.route('/api/v1/predict_form/', methods=['GET'])
def get_prediction_form():
    return render_template('predictForm.html')


"""
Test request for /api/v1/predict:
http://127.0.0.1:5000/api/v1/predict?cloud=4.0&sun=6&radiation=9.0&max_temp=12&mean_temp=10&min_temp=2&pressure=101
https://mardelmir1.pythonanywhere.com/api/v1/predict?cloud=4.0&sun=6&radiation=9.0&max_temp=12&mean_temp=10&min_temp=2&pressure=101
"""


# Predict
@app.route('/api/v1/predict/', methods=['POST', 'GET'])
def make_prediction():
    if request.method == 'POST':
        # If method == POST, get data from form
        cloud = request.form['cloud']
        sun = request.form['sun']
        radiation = request.form['radiation']
        max_temp = request.form['max_temp']
        mean_temp = request.form['mean_temp']
        min_temp = request.form['min_temp']
        pressure = request.form['pressure']
    else:
        # If method == GET, get data from the query parameters
        cloud = request.args.get('cloud')
        sun = request.args.get('sun')
        radiation = request.args.get('radiation')
        max_temp = request.args.get('max_temp')
        mean_temp = request.args.get('mean_temp')
        min_temp = request.args.get('min_temp')
        pressure = request.args.get('pressure')

    try:
        data = [cloud, sun, radiation, max_temp, mean_temp, min_temp, pressure]
        if not all(data):
            return render_template('predict.html', result=None, error='Incomplete data')

        # Convert data to float
        data = [float(x) if x != pressure else float(x) * 1000 for x in data]
        input_features = np.array([data])

        # Load model, scale data and make prediction
        model = pickle.load(open('./models/best_model.pkl', 'rb'))
        scaler = pickle.load(open('./transformers/scaler.pkl', 'rb'))
        scaled_features = scaler.transform(input_features)
        prediction = model.predict(scaled_features)[0]

        # Prepare the result as a dictionary
        result = {
            'cloud': cloud,
            'sun': sun,
            'radiation': radiation,
            'max_temp': max_temp,
            'mean_temp': mean_temp,
            'min_temp': min_temp,
            'pressure': pressure,
            'prediction': round(prediction, 2),
        }

        return render_template('predict.html', result=result)

    except ValueError as e:
        return render_template('predict.html', result=None, error=f'Value error: {str(e)}')
    except Exception as e:
        return render_template('predict.html', result=None, error=f'Unexpected error: {str(e)}')


# Update Data
@app.route('/api/v1/update_data/', methods=['POST', 'GET'])
def update_data():
    update_name = None
    if request.method == 'POST':
        f = request.files['newData']
        update_name = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], update_name))

    # List of uploaded files to select from
    data_op = u.get_file_names(UPLOAD_FOLDER)

    return render_template('updateData.html', update_name=update_name, data_op=data_op)


# Delete Data
@app.route('/api/v1/delete_data', methods=['POST'])
def delete_data():
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


# Retrain
@app.route('/api/v1/retrain/', methods=['POST', 'GET'])
def retrain_model():
    # Show original metrics
    json_url = os.path.join(SITE_ROOT, './data', 'evaluation_results.json')
    original_metrics = json.load(open(json_url))

    # List of uploaded files to select from
    data_op = u.get_file_names(UPLOAD_FOLDER)

    # Load original model and scaler
    model = pickle.load(open('./models/best_model.pkl', 'rb'))
    scaler = pickle.load(open('./transformers/scaler.pkl', 'rb'))

    new_metrics = None

    if not data_op:
        # Retrain model with default dataset
        new_model = u.retrain_model(model, scaler)
        new_metrics = new_model[0]
        pickle.dump(new_model[1], open(TEMP_MODEL_PATH, 'wb'))

    elif request.method == 'POST':
        # Process form to get selected dataset name
        dataset_name = str(request.form.getlist('dataset_name')[0])

        if dataset_name != 'default':
            # Retrain model with selected dataset
            new_data_path = f'{UPLOAD_FOLDER}/{dataset_name}'
            new_model = u.retrain_model(model, scaler, file_path=new_data_path)
        else:
            # Retrain model with default data
            new_model = u.retrain_model(model, scaler)

        new_metrics = new_model[0]
        pickle.dump(new_model[1], open(TEMP_MODEL_PATH, 'wb'))

    return render_template('retrain.html', original_metrics=original_metrics, new_metrics=new_metrics, data_op=data_op)


@app.route('/api/v1/save_model', methods=['POST'])
def save_model():
    answer = str(request.form.getlist('save_new_model')[0])
    print(answer)
    if answer == 'yes':
        # Replace the original model with the new one
        if os.path.exists(TEMP_MODEL_PATH):
            os.rename('./models/best_model.pkl', './models/original_model.pkl')
            os.rename(TEMP_MODEL_PATH, './models/best_model.pkl')
            message = 'The retrained model has been saved successfully.'
        else:
            message = 'No temporary model found. Retrain the model first.'
    else:
        # Discard the temporary model
        if os.path.exists(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)
        message = 'The retrained model was discarded.'

    return redirect(url_for('home', message=message))


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
