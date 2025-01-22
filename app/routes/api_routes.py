from flask import Blueprint, request, jsonify
from app.models.model_manager import ModelManager
import os

api_routes = Blueprint('api_routes', __name__)

model_manager = ModelManager()

@api_routes.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_path = model_manager.upload_file(file)
        return jsonify({"message": "File uploaded successfully", "filename": os.path.basename(file_path), "path": file_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_routes.route('/train', methods=['POST'])
def train_model():
    try:
        metrics = model_manager.train_model()
        return jsonify({
            "message": "Model trained successfully",
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_routes.route('/predict', methods=['POST'])
def predict():
    input_data = request.json

    try:
        prediction = model_manager.predict(input_data)
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
