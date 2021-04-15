import numpy as numpy
from flask import Flask, request, json
from joblib import dump, load


app = Flask(__name__)

@app.route('/predict')
def predict():
    features = [[request.args.get('a'),request.args.get('b'),request.args.get('c'),request.args.get('d'),request.args.get('e'),request.args.get('f'),request.args.get('g'),request.args.get('h'),request.args.get('i'),request.args.get('j'),request.args.get('k'),request.args.get('l'),request.args.get('m'),request.args.get('n'),request.args.get('o'),request.args.get('p'),request.args.get('q')]]    
    model = load('rfc.joblib')
    prediction = model.predict(features).tolist()
    predict_proba = model.predict_proba(features).tolist()

    return {"QCF":95,"Prediction":prediction, "Predict_proba":round((predict_proba)[0][0],2)}

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)
