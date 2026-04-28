import matplotlib
matplotlib.use('Agg')   # prevents matplotlib crash

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from model_loader import load_all_models
from predict import predict_all_models

app = Flask(__name__)
CORS(app)

print("Loading models...")
models = load_all_models()
print("Models loaded!")

@app.route("/")
def home():
    return "Backend is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file).convert("RGB")

        result = predict_all_models(image, models)

        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)