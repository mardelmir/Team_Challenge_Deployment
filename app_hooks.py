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

# 

if __name__ == '__main__':
    app.run()