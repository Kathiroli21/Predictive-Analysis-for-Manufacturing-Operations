import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

class ModelManager:
    def __init__(self):
        self.model = None
        self.data = None
        self.model_path = 'models/model.pkl'

    def load_data(self, filepath='data/manufacturing_data.csv'):
        self.data = pd.read_csv(filepath)

    def train_model(self):
        if self.data is None:
            self.load_data()

        X = self.data[['Temperature', 'Run_Time']]
        y = self.data['Downtime_Flag']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        joblib.dump(self.model, self.model_path)
        return metrics

    def predict(self, input_data):
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise Exception("Model not trained yet.")

        features = [[input_data['Temperature'], input_data['Run_Time']]]
        prediction = self.model.predict(features)[0]
        confidence = max(self.model.predict_proba(features)[0])

        return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}
