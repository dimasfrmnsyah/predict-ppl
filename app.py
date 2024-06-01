import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from diabetes_prediction import ANN

app = Flask(__name__)
filepath_scaler = 'diabetes_scaler.joblib'
filepath_model = 'diabetes_model.h5'
ann_model = ANN(loc_scaler = filepath_scaler, loc_model = filepath_model)


@app.route('/predict',methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    score, prediction = ann_model.klasifikasi(final_features)

    if prediction == 'Diabetes':
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 'Sehat':
        pred = "You don't have Diabetes."
    output = pred
    return jsonify(prediction=output,score=str(score))


if __name__ == "__main__":
    app.run(debug=True, port=3000)
