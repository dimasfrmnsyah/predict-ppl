import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Kelas untuk prediksi menggunakan ANN yang sudah dilatih
class ANN():
  def __init__ (self, loc_scaler, loc_model):
    self.diabetes_scaler = joblib.load(loc_scaler)
    self.diabetes_model = load_model(loc_model)

  def klasifikasi(self, data):
    data_scaled = self.diabetes_scaler.transform(data)
    predicted_score = self.diabetes_model.predict(data_scaled, verbose=0)
    round_score = np.round(predicted_score)
    if round_score == 1:
      predicted_class = 'Diabetes'
    else:
      predicted_class = 'Sehat'
    return predicted_score[0][0], predicted_class