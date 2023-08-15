# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:50:04 2023

@author: Anas
"""

from flask import Flask, request, render_template
import numpy as np

import pickle
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predic_house_price():
    rm = request.form.get('rm')
    zn = request.form.get('zn')
    dis= request.form.get('dis')
    
    result = model.predict(np.array([rm,zn,dis]).reshape(1,3))[0]
    result = "${:.2f}".format(result)
    
    # return str(result) # this is printing on new page
    return render_template("index.html", result=result)
    
    
    


if __name__ == '__main__':
    app.run(debug=True)