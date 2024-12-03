import os
import pickle
import subprocess

import numpy as np
import pandas as pd

from flask import Flask, render_template, jsonify, request

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True


# Landing page (endpoint /)
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

# Predict
@app.route('/api/v1/predict/', methods = ['POST', 'GET'])
def make_prediction():
    prediction = None
    if request.method == 'POST':
        
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # prediction = model.predict(np.array([[feature1, feature2]])) 

    return render_template('predict.html', prediction = prediction)
        
# Forecast
@app.route('/api/v1/forecast/', methods = ['POST', 'GET'])
def forecast():
    prediction = None
    if request.method == 'POST':
        
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # prediction = model.predict(np.array([[feature1, feature2]])) 

    return render_template('forecast.html', prediction = prediction)

# Udate Data
@app.route('/api/v1/update_data/', methods = ['POST', 'GET'])
def update_data():
    prediction = None
    if request.method == 'POST':
        
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # prediction = model.predict(np.array([[feature1, feature2]])) 

    return render_template('updateData.html', prediction = prediction)

# Retrain
@app.route('/api/v1/retrain/', methods = ['POST', 'GET'])
def retrain_model():
    prediction = None
    if request.method == 'POST':
        
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # prediction = model.predict(np.array([[feature1, feature2]])) 

    return render_template('retrain.html', prediction = prediction)



if __name__ == '__main__':
    app.run()