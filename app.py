# app.py (Flask Backend)
from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('visualize.html')

@app.route('/api/points')
def get_points():
    df = pd.read_csv('uploads/house_price.csv')
    # One-hot encode 'city'
    df_encoded = pd.get_dummies(df, columns=['city'])

    # Define input and output features
    input_features = ['floors', 'bedrooms', 'area_sqft'] + [col for col in df_encoded.columns if col.startswith('city_')]
    output_feature = 'price_in_lakhs'

    # Normalize inputs and output
    scaler_in = MinMaxScaler()
    X_normalized = scaler_in.fit_transform(df_encoded[input_features])

    scaler_out = MinMaxScaler()
    y_normalized = scaler_out.fit_transform(df_encoded[[output_feature]])

    n = len(input_features)
    angles = [i * (2 * np.pi / n) for i in range(n)]

    all_points = []
    lines = []

    for row, price in zip(X_normalized, y_normalized):
        row_points = []
        y_val = price[0]

        for val, angle in zip(row, angles):
            x = val * np.cos(angle)
            z = val * np.sin(angle)
            point = (x, y_val, z)
            row_points.append(point)
            lines.append([(0, y_val, 0), point])

        all_points.append(row_points)

    return jsonify({
        'points': all_points,
        'lines': lines,
        'labels': input_features
    })

if __name__ == '__main__':
    app.run(debug=True)
