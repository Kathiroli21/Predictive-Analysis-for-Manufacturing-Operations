import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from werkzeug.utils import secure_filename
import time


class ModelManager:
    def __init__(self):
        self.model = None
        self.data = None
        self.model_path = 'models/model.pkl'
        self.BASE_UPLOAD_FOLDER = 'data'
        self.MODEL_FOLDER = 'models'
        os.makedirs(self.BASE_UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.MODEL_FOLDER, exist_ok=True)

    def load_data(self, filepath='data/manufacturing_data.csv'):
        self.data = pd.read_csv(filepath)
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise Exception("Model not trained yet.")

    def upload_file(self, file):
            # Ensure the upload folder exists
            if not os.path.exists(self.BASE_UPLOAD_FOLDER):
                os.makedirs(self.BASE_UPLOAD_FOLDER)

            # Generate a unique filename with timestamp
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            unique_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"

            # Save the file
            file_path = os.path.join(self.MODEL_FOLDER , unique_filename)
            file.save(file_path)

            return file_path

    def train_model(self):
        if not os.path.exists(self.BASE_UPLOAD_FOLDER):
            raise Exception("No files found")

        files = os.listdir(self.BASE_UPLOAD_FOLDER)
        if not files:
            raise Exception("No files uploaded yet")

        latest_file = max([os.path.join(self.BASE_UPLOAD_FOLDER, f) for f in files], key=os.path.getctime)
        
        try:
            data = pd.read_csv(latest_file)
        except Exception as e:
            raise Exception(f"Error reading CSV: {str(e)}")

        required_columns = {'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Downtime'}
        if not required_columns.issubset(data.columns):
            raise Exception(f"Dataset must contain columns: {required_columns}")

        data['Downtime'] = data['Downtime'].map({'Machine_Failure': 1, 'No_Machine_Failure': 0})
        
        features = [
            'Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)',
            'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
            'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)', 'Voltage(volts)', 
            'Torque(Nm)', 'Cutting(kN)'
        ]
        
        if not set(features).issubset(data.columns):
            raise Exception(f"Dataset must contain required features: {features}")

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

        model_path = os.path.join(self.MODEL_FOLDER, 'model.pkl')
        joblib.dump(best_model, model_path)

        return {
            "latest_file":latest_file,
            "accuracy": accuracy,
            "f1_score": f1
        }

    def predict(self, input_data):
        if self.model is None:
            self.load_model()
        
        expected_features = [
            'Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)',
            'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
            'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)', 'Voltage(volts)',
            'Torque(Nm)', 'Cutting(kN)'
        ]
        
        if not input_data:
            raise ValueError("Invalid input. Please provide feature values.")
        
        missing_features = [feature for feature in expected_features if feature not in input_data]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[expected_features]
        except Exception as e:
            raise ValueError(f"Error processing input data: {str(e)}")
        
        prediction = self.model.predict(input_df)[0]
        confidence = max(self.model.predict_proba(input_df)[0])
        
        return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}