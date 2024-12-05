import os
import pickle
import subprocess

import numpy as np
import pandas as pd

from flask import Flask, render_template, jsonify, request, url_for, redirect

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True


# Landing page (endpoint /)
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')


# Predict Form
@app.route('/api/v1/predict_form/', methods = ['GET'])
def get_prediction_form():
    return render_template('predictForm.html')
    

'''
Test request for /api/v1/predict:
http://127.0.0.1:5000/api/v1/predict?pressure=15&sun=60&mean_temp=80
'''

# Predict
@app.route('/api/v1/predict/', methods = ['POST', 'GET'])
def make_prediction():
    if request.method == 'POST':
        # Form data
        pressure = request.form.get('pressure')
        sun = request.form.get('sun')
        mean_temp = request.form.get('mean_temp')
        
        # This is a test to see that it retrieves info from form correctly, the prediction would go here instead
        prediction_result = f'''<h3>Form parameters:</h3> 
            <p>pressure: {pressure}</p>
            <p>sun: {sun}</p>
            <p>mean_temp: {mean_temp}</p>
        '''
        
        # Redirection
        return redirect(url_for('make_prediction', result = prediction_result))
    
    # If method = GET, gets data from query parameters
    result = request.args.get('result', None)
    
    # Renders template with result
    return render_template('predict.html', result = result)

        
        
# Forecast
@app.route('/api/v1/forecast/', methods = ['POST', 'GET'])
def forecast():
    forecast = None
    if request.method == 'POST':
 
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # prediction = model.predict(np.array([[feature1, feature2]])) 

    return render_template('forecast.html', forecast = forecast)


# MARÍA
# Udate Data
@app.route('/api/v1/update_data/', methods = ['POST', 'GET'])
def update_data():
    if request.method == 'POST':
        pass
    return render_template('updateData.html')


# Retrain
@app.route('/api/v1/retrain/', methods = ['POST', 'GET'])
def retrain_model():
    prediction = None
    if request.method == 'POST':
        
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # prediction = model.predict(np.array([[feature1, feature2]])) 

    return render_template('retrain.html', prediction = prediction)


# Webhook
@app.route('/webhook', methods = ['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = '/home/mardelmir1/Team_Challenge_Deployment'
    servidor_web = '/var/www/mardelmir1_pythonanywhere_com_wsgi.py'

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']

            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe!'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull', clone_url], check = True)
                subprocess.run(['touch', servidor_web], check = True)  # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400

if __name__ == '__main__':
    app.run()