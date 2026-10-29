import os
import io
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, render_template, jsonify
from PIL import Image
import joblib
from disease_info import DISEASE_INFO

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    print("Warning: flask_cors not installed. Running without CORS.")

# Load best model and class names
print("Loading model and class names...")
best_model = None
class_names = []

try:
    best_model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    class_names_path = os.path.join(os.path.dirname(__file__), "class_names.pkl")
    
    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)
    else:
        print(f"Warning: {best_model_path} not found.")

    if os.path.exists(class_names_path):
        class_names = joblib.load(class_names_path)
    else:
        print(f"Warning: {class_names_path} not found.")
        
    print(f"Loaded model successfully. Classes: {class_names}")
except Exception as e:
    print(f"Warning: Models not loaded yet. Error: {e}")

# Load PyTorch MobileNetV2 Feature Extractor
print("Loading PyTorch Feature Extractor...")
try:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    base_model = models.mobilenet_v2(weights=weights)
    feature_extractor = base_model.features
    feature_extractor.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("MobileNetV2 feature extractor initialized.")
except Exception as e:
    print(f"Error loading MobileNetV2: {e}")
    feature_extractor = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": best_model is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "classes": class_names
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please upload an image file with key 'file'."}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    if not best_model or not feature_extractor:
        return jsonify({
            "error": "Model or feature extractor is still loading or failed to load. Please check server logs."
        }), 503

    try:
        # Read and validate image
        img_bytes = file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "The uploaded file is not a valid image format (PNG, JPG, JPEG supported)."}), 400
        
        # Preprocess
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(input_batch)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        
        features_np = features.cpu().numpy()
        
        # Predict using trained classifier
        pred_idx = best_model.predict(features_np)[0]
        predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        
        # Calculate confidence
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(features_np)[0]
            confidence = float(np.max(probs))
        else:
            confidence = 1.0 # fallback
            
        # Retrieve disease details
        info = DISEASE_INFO.get(predicted_class, {
            "edible": False, 
            "description": "Unknown condition or unlisted pomegranate defect.", 
            "health_effects": "Consult an agricultural expert before consumption."
        })
        
        return jsonify({
            "class": predicted_class,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "edible": info["edible"],
            "description": info["description"],
            "health_effects": info["health_effects"]
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Server processing error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    print(f"Starting Pomegranate Disease Detection server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
