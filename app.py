from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Define dataset configurations
DATASETS = {
    'house_price': {
        'file': 'uploads/house_price.csv',
        'input_features': ['floors', 'bedrooms', 'area_sqft'],
        'categorical_features': ['city'],
        'output_feature': 'price_in_lakhs'
    },
    'titanic': {
        'file': 'uploads/titanic.csv',
        'input_features': ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
        'categorical_features': ['Sex', 'Embarked'],
        'output_feature': 'Survived'
    }
}

@app.route('/')
def index():
    return render_template('visualize.html')

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Return a list of available datasets for the dropdown."""
    return jsonify({
        'datasets': list(DATASETS.keys())
    })

@app.route('/api/points', methods=['GET'])
def get_points():
    """Fetch and process data points for the selected dataset."""
    # Get the dataset name from query parameter
    dataset_name = request.args.get('dataset', 'house_price')  # Default to house_price
    if dataset_name not in DATASETS:
        return jsonify({'error': f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASETS.keys())}"}), 400

    # Load dataset configuration
    config = DATASETS[dataset_name]
    file_path = config['file']
    input_features = config['input_features']
    categorical_features = config['categorical_features']
    output_feature = config['output_feature']

    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return jsonify({'error': f"File '{file_path}' not found."}), 404
    except Exception as e:
        return jsonify({'error': f"Error loading dataset: {str(e)}"}), 500

    # Handle missing values (simple imputation)
    for col in input_features:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    for col in categorical_features:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df[output_feature].fillna(df[output_feature].mean(), inplace=True)

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Update input features to include one-hot encoded columns
    encoded_features = [col for col in df_encoded.columns if any(col.startswith(f"{cat}_") for cat in categorical_features)]
    all_input_features = input_features + encoded_features

    # Limit to 6 features as per SRS
    all_input_features = all_input_features[:6]

    # Normalize inputs and output
    scaler_in = MinMaxScaler()
    try:
        X_normalized = scaler_in.fit_transform(df_encoded[all_input_features])
    except KeyError as e:
        return jsonify({'error': f"Missing feature in dataset: {str(e)}"}), 400

    scaler_out = MinMaxScaler(feature_range=(0, 1))
    y_normalized = scaler_out.fit_transform(df_encoded[[output_feature]])

    # Calculate angles for radial layout
    n = len(all_input_features)
    if n == 0:
        return jsonify({'error': 'No input features available for visualization.'}), 400
    angles = [i * (2 * np.pi / n) for i in range(n)]

    # Transform data into radial coordinates
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
        'labels': all_input_features,
        'output_label': output_feature
    })

if __name__ == '__main__':
    app.run(debug=True)