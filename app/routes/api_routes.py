from flask import Blueprint, request, jsonify
import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder


# Define the blueprint
api_bp = Blueprint('api_routes', __name__)

# Base directories
BASE_UPLOAD_FOLDER = 'data'
MODEL_FOLDER = 'models'
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Endpoint: Upload
@api_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"

        
        file_path = os.path.join(BASE_UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        return jsonify({"message": "File uploaded successfully", "filename": unique_filename, "path": file_path}), 200


@api_bp.route('/train', methods=['POST'])
def train_model():
    if not os.path.exists(BASE_UPLOAD_FOLDER):
        return jsonify({"error": "No files found"}), 400

    files = os.listdir(BASE_UPLOAD_FOLDER)
    if not files:
        return jsonify({"error": "No files uploaded yet"}), 400

    latest_file = max([os.path.join(BASE_UPLOAD_FOLDER, f) for f in files], key=os.path.getctime)
    
    try:
        data = pd.read_csv(latest_file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400

    required_columns = {'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Downtime'}
    if not required_columns.issubset(data.columns):
        return jsonify({"error": f"Dataset must contain columns: {required_columns}"}), 400

    data['Downtime'] = data['Downtime'].map({'Machine_Failure': 1, 'No_Machine_Failure': 0})
    
    features = [
        'Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)',
        'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
        'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)', 'Voltage(volts)', 
        'Torque(Nm)', 'Cutting(kN)'
    ]
    
    if not set(features).issubset(data.columns):
        return jsonify({"error": f"Dataset must contain required features: {features}"}), 400

    X = data[features]
    y = data['Downtime']

    X.fillna(X.median(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    model_path = os.path.join(MODEL_FOLDER, 'optimized_model.pkl')
    joblib.dump(best_model, model_path)

    return jsonify({  
        "accuracy": accuracy,
        "f1_score": f1
    }), 200
@api_bp.route('/predict', methods=['POST'])
def predict():
    model_path = os.path.join(MODEL_FOLDER, 'optimized_model.pkl')

    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found. Train the model first."}), 400

    
    model = joblib.load(model_path)

    
    expected_features = [
        'Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)',
        'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
        'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)', 'Voltage(volts)',
        'Torque(Nm)', 'Cutting(kN)'
    ]

    input_data = request.json

    
    if not input_data:
        return jsonify({"error": "Invalid input. Please provide feature values."}), 400

    missing_features = [feature for feature in expected_features if feature not in input_data]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400

    
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[expected_features]  
    except Exception as e:
        return jsonify({"error": f"Error processing input data: {str(e)}"}), 400

    prediction = model.predict(input_df)[0]
    confidence = max(model.predict_proba(input_df)[0])

    return jsonify({
        
        "Downtime": "Yes" if prediction == 1 else "No",
        "Confidence": round(confidence, 2)
    }), 200
