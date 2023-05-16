#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:34:42 2022

@author: shyam
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os
import requests
import json

app = Flask(__name__)
model = pickle.load(open('Visarf.pkl', 'rb'))




@app.route('/')
def home():
    return render_template('Visa_Approval.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    print(input_features)
    features_value = [np.array(input_features)]
    print(features_value)
    
    
    
    features_name = ['FULL_TIME_POSITION', 'PREVAILING_WAGE', 'YEAR','SOC_N']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    output=np.argmax(output)  
    print(output)
        

    return render_template('resultVA.html', prediction_text=output)

if __name__ == '__main__':
  
    app.run(debug=True)
