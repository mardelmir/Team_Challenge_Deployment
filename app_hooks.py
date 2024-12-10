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
http://127.0.0.1:5000/api/v1/predict?cloud=4.0&sun=6&radiation=9.0&max_temp=12&mean_temp=10&min_temp=2&pressure=101&snow=2.0
"""


# Predict
@app.route('/api/v1/predict/', methods=['POST', 'GET'])
def make_prediction():
    if request.method == 'POST':
        # Form data
        cloud = request.form['cloud']
        sun = request.form['sun']
        radiation = request.form['radiation']
        max_temp = request.form['max_temp']
        mean_temp = request.form['mean_temp']
        min_temp = request.form['min_temp']
        pressure = float(request.form['pressure']) * 1000

        # Redirection
        return redirect(
            url_for(
                'make_prediction',
                cloud=cloud,
                sun=sun,
                radiation=radiation,
                max_temp=max_temp,
                mean_temp=mean_temp,
                min_temp=min_temp,
                pressure=pressure,
            )
        )

    # If method = GET, get data from the query parameters
    cloud = request.args.get('cloud', None)
    sun = request.args.get('sun', None)
    radiation = request.args.get('radiation', None)
    max_temp = request.args.get('max_temp', None)
    mean_temp = request.args.get('mean_temp', None)
    min_temp = request.args.get('min_temp', None)
    pressure = request.args.get('pressure', None)

    data = [cloud, sun, radiation, max_temp, mean_temp, min_temp, pressure]
    data = [float(arg) for arg in data if arg is not None]
    result = None

    if all(data):
        input_features = np.array([data])

        # Put the input data through the scaler used to produce the model before making prediction
        model = pickle.load(open('./models/best_model.pkl', 'rb'))
        scaler = pickle.load(open('./transformers/scaler.pkl', 'rb'))
        scaled_features = scaler.transform(input_features)
        prediction = model.predict(scaled_features)[0]

        # Prepare the result as a dictionary
        result = (
            {
                'cloud': cloud,
                'sun': sun,
                'radiation': radiation,
                'max_temp': max_temp,
                'mean_temp': mean_temp,
                'min_temp': min_temp,
                'pressure': pressure,
                'prediction': round(prediction, 2),
            }
            if all(data)
            else None
        )

    # Render template with result
    return render_template('predict.html', result=result)


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
    json_url = os.path.join(SITE_ROOT, './data', 'evaluation_results.json')
    original_metrics = json.load(open(json_url))

    # dataset_name = 'original'

    # if request.method == 'POST':
    #     dataset_name = str(request.form.getlist('dataset_name')[0])
    #     print(dataset_name)

    # Review
    # List of uploaded files to select from
    data_op = u.get_file_names(UPLOAD_FOLDER)

    model = pickle.load(open('./models/best_model.pkl', 'rb'))
    new_model = u.retrain_model(model)

    return render_template('retrain.html', original_metrics=original_metrics, new_metrics=new_model[0], data_op=data_op)


@app.route('/api/v1/save_model', methods=['POST'])
def save_model():
    pass


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
