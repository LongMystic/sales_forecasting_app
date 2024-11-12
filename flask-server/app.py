from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import timedelta

app = Flask(__name__)

# Load or define model and data
# For simplicity, use a placeholder model and data here
n_steps = 3  # Ensure this matches the model's expected input

# Load the model (or train if needed)
# model = load_model('path_to_your_trained_model.h5')

# Dummy data for demonstration
data = pd.DataFrame({
    'day': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'sales': [4, 6, 5, 7, 8, 5, 6, 7, 5, 8]
})
scaler = MinMaxScaler(feature_range=(0, 1))
data['scaled_sales'] = scaler.fit_transform(data[['sales']])

# Generate predictions
def get_predictions(last_sequence, steps=2):
    predictions = []
    for _ in range(steps):
        # Predict next sales value
        # prediction = model.predict(last_sequence.reshape(1, n_steps, 1))
        prediction = np.array([[0.5]])  # Dummy prediction for placeholder
        predictions.append(prediction[0][0])
        last_sequence = np.append(last_sequence[1:], prediction)  # Update sequence
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

@app.route('/train-data', methods=['GET'])
def get_train_data():
    return jsonify(data[['day', 'sales']].to_dict(orient='records'))

@app.route('/predict', methods=['GET'])
def predict_sales():
    last_sequence = data['scaled_sales'].values[-n_steps:]
    predictions = get_predictions(last_sequence)
    prediction_dates = [data['day'].iloc[-1] + timedelta(days=i+1) for i in range(len(predictions))]
    prediction_data = [{'day': date.strftime('%Y-%m-%d'), 'sales': float(sale)} for date, sale in zip(prediction_dates, predictions)]
    return jsonify(prediction_data)

if __name__ == '__main__':
    app.run(debug=True)
