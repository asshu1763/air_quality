from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('majorproject.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        SO2 = float(request.form['SO2'])
        NO2=float(request.form['NO2'])
        Particulate_Matter=float(request.form['Particulate_Matter'])
        CO=float(request.form['CO'])
        O3=float(request.form['O3'])
        BTX=float(request.form['BTX'])
        prediction=model.predict([[SO2,NO2,Particulate_Matter,CO,O3,BTX]])
        output=prediction
        return render_template('index.html',prediction_text="The Air Quality is {}".format(output[0]))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
