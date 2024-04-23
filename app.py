from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
import pickle
import os

app = Flask(__name__)
crop_model=pickle.load(open(r'C:\Users\girfana\Downloads\Crop\Crop\Models\RF.pkl', 'rb'))
fer_model=pickle.load(open(r'C:\Users\girfana\Downloads\Crop\Crop\Models\FR.pkl', 'rb'))



@app.route('/') # rendering the html template 
def home():
    return render_template("index.html")

@ app.route('/croprecom.html')
def crop_recommend():
    title = 'crop-recommend - Crop Recommendation'
    return render_template('croprecom.html', title=title)

@ app.route('/fertilizer.html')
def fertilizer_recommendation():
    title = '- Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)


#CROP RECOMMENDATION
@app.route('/submit1',methods=["POST", "GET"])# route to show the predictions in a web UI 
def submit(): 
    input_feature=[float(x) for x in request.form.values()]
    input_feature=[np.array(input_feature)]
    print(input_feature)
    
    # predictions using the loaded model file
    prediction=crop_model.predict(input_feature)
    print(prediction)
    print(type(prediction))
    return render_template("output.html", result = prediction[0])
    
#FERTILIZER RECOMMENDATION
@app.route('/submit2',methods=["POST", "GET"])# route to show the predictions in a web UI 
def submit2():
    ct={'Barley':1, 'Cotton':1, 'Ground Nuts':2, 'Maize':3, 'Millets':4, 'Oil seeds':5, 
        'Paddy':6, 'Pulses':7, 'Sugarcane':8, 'Tobacco':9, 'Wheat':10}
    st={'Black':0, 'Clayey':1, 'Loamy':2, 'Red':3, 'Sandy':4}
    l=[] 
    t = int(request.form['temperature'])
    l.append(t)
    h = int(request.form['humidity'])
    l.append(h)
    sm = int(request.form['soilmoisture'])
    l.append(sm)
    s=request.form['soiltype']
    l.append(st[s])
    N = int(request.form['nitrogen'])
    l.append(N)
    K = int(request.form['pottasium'])
    l.append(K)
    P = int(request.form['phosphorous'])
    l.append(P)
    c = request.form['cropname']
    l.append(ct[c])
    input_feature=[np.array(l)]
    prediction=fer_model.predict(input_feature)
    ft={0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}
    print(prediction)
    return render_template("output.html", result = ft[prediction[0]])

if __name__=="__main__":
    # app.run(host= '0.0.0.0', port=8000, debug=True)
    port=int(os.environ.get("PORT " ,5000))
    app.run(debug=True)
