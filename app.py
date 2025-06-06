from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/house_price')
def house_price_eda():
    return render_template('house_price_eda.html')

@app.route('/titanic_eda')
def titanic_eda():
    return render_template('titanic_eda.html')

@app.route('/breast_cancer_eda')
def breast_cancer_eda():
    return render_template('breast_cancer_eda.html')

@app.route('/food_delivery_times')
def delivery_time_eda():
    return render_template('delivery_time_eda.html')

# Define dataset configurations
DATASETS = {
    'house_price': {
        'file': 'uploads/house_price.csv',
        'input_features': ['floors', 'bedrooms', 'area_sqft'],
        'categorical_features': ['city'],
        'output_feature': 'price_in_lakhs'
    },
    'titanic': {
        'file': 'uploads/tested.csv',
        'input_features': ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
        'categorical_features': ['Sex', 'Embarked'],
        'output_feature': 'Survived'
    },
    'breast_cancer': {
        'file': 'uploads/breast_cancer.csv',
        'input_features': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'],
        'categorical_features': [],
        'output_feature': 'diagnosis'
    },
    'food_delivery_times': {
        'file': 'uploads/food_delivery_times.csv',
        'input_features': ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs'],
        'categorical_features': ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'],
        'output_feature': 'Delivery_Time_min'
    },
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

def preprocess_data(df, config):
    """Enhanced data preprocessing with better error handling"""
    required_columns = config['input_features'] + config['categorical_features'] + [config['output_feature']]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    df_processed = df.copy()
    
    for col in config['input_features']:
        if df_processed[col].dtype in [np.float64, np.int64]:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0)
    
    for col in config['categorical_features']:
        if df_processed[col].mode().empty:
            df_processed[col] = df_processed[col].fillna('Unknown')
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    if df_processed[config['output_feature']].dtype in [np.float64, np.int64]:
        df_processed[config['output_feature']] = df_processed[config['output_feature']].fillna(df_processed[config['output_feature']].median())
    else:
        df_processed[config['output_feature']] = df_processed[config['output_feature']].fillna(df_processed[config['output_feature']].mode()[0])
    
    return df_processed

@app.route('/api/eda', methods=['GET'])
def get_eda():
    dataset_name = request.args.get('dataset', 'house_price')
    if dataset_name not in DATASETS:
        return jsonify({'error': 'Dataset not found'}), 400

    config = DATASETS[dataset_name]
    try:
        df = pd.read_csv(config['file'])
        print(f"Loaded {dataset_name} dataset with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        return jsonify({'error': f"File '{config['file']}' not found."}), 404
    except Exception as e:
        return jsonify({'error': f"Error loading dataset: {str(e)}"}), 500

    try:
        df_processed = preprocess_data(df, config)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Error preprocessing data: {str(e)}"}), 500

    # Feature statistics
    feature_stats = {}
    for col in config['input_features'] + [config['output_feature']]:
        if df_processed[col].dtype in [np.float64, np.int64]:
            feature_stats[col] = {
                'type': 'numerical',
                'mean': float(df_processed[col].mean()),
                'std': float(df_processed[col].std()),
                'min': float(df_processed[col].min()),
                'max': float(df_processed[col].max()),
                'missing': int(df[col].isna().sum())  # Use original df for missing count
            }
        else:
            feature_stats[col] = {
                'type': 'categorical',
                'unique_values': int(df_processed[col].nunique()),
                'most_common': str(df_processed[col].mode()[0]) if not df_processed[col].mode().empty else 'N/A',
                'missing': int(df[col].isna().sum())  # Use original df for missing count
            }

    # Radial wheel graph insights
    n = len(config['input_features'])
    if n > 0:
        angles = [2 * np.pi * i / n for i in range(n)]
        coords = []
        
        for _, row in df_processed.iterrows():
            try:
                x = sum(float(row[f]) * np.cos(a) for f, a in zip(config['input_features'], angles))
                z = sum(float(row[f]) * np.sin(a) for f, a in zip(config['input_features'], angles))
                y = float(row[config['output_feature']])
                coords.append([x, z, y])
            except (ValueError, TypeError) as e:
                print(f"Error processing row: {e}")
                continue
        
        if coords:
            coords = np.array(coords)
            output_values = coords[:, 2]
            
            # Calculate quantiles safely
            q40 = np.percentile(output_values, 40)
            q70 = np.percentile(output_values, 70)
            
            tier_counts = {
                'Low (< 40%)': int(np.sum(output_values < q40)),
                'Mid (40-70%)': int(np.sum((output_values >= q40) & (output_values < q70))),
                'High (> 70%)': int(np.sum(output_values >= q70))
            }
            
            graph_insights = (
                f"Distribution across tiers in the radial wheel graph:\n"
                f"- Low (< 40%): {tier_counts['Low (< 40%)']} points\n"
                f"- Mid (40-70%): {tier_counts['Mid (40-70%)']} points\n"
                f"- High (> 70%): {tier_counts['High (> 70%)']} points\n"
                f"This indicates how data points are spread across the output feature's range when projected radially."
            )
        else:
            graph_insights = "Unable to generate radial wheel insights due to data processing issues."
    else:
        graph_insights = "No input features available for radial wheel visualization."

    return jsonify({
        'dataset': dataset_name,
        'num_data_points': len(df_processed),
        'features': feature_stats,
        'output_feature': config['output_feature'],
        'graph_insights': graph_insights
    })

@app.route('/api/points', methods=['GET'])
def get_points():
    """Fetch and process data points for the selected dataset."""
    dataset_name = request.args.get('dataset', 'house_price')
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
        print(f"Loading {dataset_name}: Shape {df.shape}, Columns: {list(df.columns)}")
    except FileNotFoundError:
        return jsonify({'error': f"File '{file_path}' not found."}), 404
    except Exception as e:
        return jsonify({'error': f"Error loading dataset: {str(e)}"}), 500

    try:
        df_processed = preprocess_data(df, config)
    except Exception as e:
        return jsonify({'error': f"Error preprocessing data: {str(e)}"}), 500

    # One-hot encode categorical features
    try:
        df_encoded = pd.get_dummies(df_processed, columns=categorical_features, dummy_na=False)
        print(f"After encoding: {df_encoded.shape}, Columns: {list(df_encoded.columns)}")
    except Exception as e:
        return jsonify({'error': f"Error encoding categorical features: {str(e)}"}), 500

    # Update input features to include one-hot encoded columns
    encoded_features = []
    for cat in categorical_features:
        encoded_cols = [col for col in df_encoded.columns if col.startswith(f"{cat}_")]
        encoded_features.extend(encoded_cols)
    
    all_input_features = input_features + encoded_features

    # Limit to 6 features as per SRS
    all_input_features = all_input_features[:6]
    print(f"Final input features: {all_input_features}")

    # Check if all features exist
    missing_features = [f for f in all_input_features if f not in df_encoded.columns]
    if missing_features:
        return jsonify({'error': f"Missing features after encoding: {missing_features}"}), 400

    # Normalize inputs and output
    scaler_in = MinMaxScaler()
    scaler_out = MinMaxScaler(feature_range=(0, 1))
    
    try:
        X_normalized = scaler_in.fit_transform(df_encoded[all_input_features])
        y_normalized = scaler_out.fit_transform(df_encoded[[output_feature]])
        print(f"Normalization successful: X shape {X_normalized.shape}, y shape {y_normalized.shape}")
    except Exception as e:
        return jsonify({'error': f"Error during normalization: {str(e)}"}), 500

    # Calculate angles for radial layout
    n = len(all_input_features)
    if n == 0:
        return jsonify({'error': 'No input features available for visualization.'}), 400
    
    angles = [i * (2 * np.pi / n) for i in range(n)]

    # Transform data into radial coordinates
    all_points = []
    lines = []
    
    try:
        for row, price in zip(X_normalized, y_normalized):
            row_points = []
            y_val = float(price[0])

            for val, angle in zip(row, angles):
                x = float(val) * np.cos(angle)
                z = float(val) * np.sin(angle)
                point = [x, y_val, z]
                row_points.append(point)
                lines.append([[0, y_val, 0], point])

            all_points.append(row_points)
        
        print(f"Generated {len(all_points)} point sets with {len(lines)} lines")
        
    except Exception as e:
        return jsonify({'error': f"Error generating visualization coordinates: {str(e)}"}), 500

    return jsonify({
        'points': all_points,
        'lines': lines,
        'labels': all_input_features,
        'output_label': output_feature
    })

@app.route('/eda')
def eda_page():
    return render_template('eda.html')

if __name__ == '__main__':
    app.run(debug=True)