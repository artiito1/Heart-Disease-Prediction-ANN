
import numpy as np
import pandas as pd
from sklearn import preprocessing
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('models/model0.pkl', 'rb'))
data = pd.read_csv("Heart Disease dataset.csv")
features=data.drop(["target"],axis=1)
col_names = list(features.columns)
s_scaler = preprocessing.StandardScaler()
features_df = pd.DataFrame(data, columns=col_names) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    newFeatures = np.array(int_features).reshape(1, -1) 
    newFeatures_df = pd.DataFrame(newFeatures, columns=['age', 'sex','cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
     'slope', 'ca', 'thal'])
    fullFeatures_df = pd.concat([newFeatures_df,features_df],axis=0)
    fullFeatures_df = s_scaler.fit_transform(fullFeatures_df)
    prediction = model.predict(fullFeatures_df)  

    output = np.round(prediction[0][0]*100,3)

    return render_template('index.html', prediction_text='Kalp hastalığı olasılığı {}'.format(output))


if __name__ == "__main__":
    app.run()