import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'skin_cancer_model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load class names
def load_class_names():
    class_names = []
    try:
        with open(os.path.join(BASE_DIR, 'class_names.txt'), 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        # If file doesn't exist, use directory names from training data
        train_dir = os.path.join(BASE_DIR, 'Skin cancer DATA', 'Train')
        class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    return class_names

# Load the model
def get_model():
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# Make prediction
def predict_image(img_path, model, class_names):
    try:
        processed_image = preprocess_image(img_path)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # Get top 3 predictions
        top_indices = predictions[0].argsort()[-3:][::-1]
        top_predictions = [
            {"class": class_names[i], "confidence": float(predictions[0][i])}
            for i in top_indices
        ]
        
        return {
            "success": True,
            "predicted_class": class_names[predicted_class_index],
            "confidence": confidence,
            "top_predictions": top_predictions
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        model = get_model()
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"})
        
        class_names = load_class_names()
        result = predict_image(file_path, model, class_names)
        
        # Add the image path to the result
        if result["success"]:
            result["image_path"] = os.path.join('uploads', filename)
        
        return jsonify(result)
    
    return jsonify({"success": False, "error": "File type not allowed"})

@app.route('/predict-features', methods=['POST'])
def predict_features():
    """
    Endpoint for feature-based skin cancer prediction
    """
    try:
        # Get feature data from request
        feature_data = request.json
        
        if not feature_data:
            return jsonify({"success": False, "error": "No feature data provided"})
        
        # In a real application, you would use these features with a trained model
        # For now, we'll implement a simple rule-based system
        
        # Extract features
        color = feature_data.get('color', '')
        size = feature_data.get('size', '')
        shape = feature_data.get('shape', '')
        texture = feature_data.get('texture', '')
        evolution = feature_data.get('evolution', '')
        
        # Simple rule-based risk assessment
        risk_score = 0
        max_score = 0
        
        # Color analysis
        if color:
            max_score += 30
            if color in ['black', 'multi']:
                risk_score += 30  # High risk
            elif color in ['blue', 'red']:
                risk_score += 20  # Medium risk
            else:
                risk_score += 10  # Lower risk
        
        # Size analysis
        if size:
            max_score += 25
            try:
                size_num = float(size)
                if size_num > 6:
                    risk_score += 25  # High risk
                elif size_num > 3:
                    risk_score += 15  # Medium risk
                else:
                    risk_score += 5   # Lower risk
            except ValueError:
                pass
        
        # Shape analysis
        if shape:
            max_score += 20
            if shape in ['asymmetric', 'irregular']:
                risk_score += 20  # High risk
            else:
                risk_score += 5   # Lower risk
        
        # Texture analysis
        if texture:
            max_score += 15
            if texture == 'ulcerated':
                risk_score += 15  # High risk
            elif texture in ['crusty', 'scaly']:
                risk_score += 10  # Medium risk
            else:
                risk_score += 5   # Lower risk
        
        # Evolution analysis
        if evolution:
            max_score += 10
            if evolution == 'yes':
                risk_score += 10  # High risk
            else:
                risk_score += 2   # Lower risk
        
        # Calculate percentage
        percentage = (risk_score / max_score) * 100 if max_score > 0 else 0
        
        # Determine prediction based on risk score
        if percentage > 70:
            prediction = "Melanoma"
            confidence = percentage / 100
            other_predictions = [
                {"class": "Melanoma", "confidence": confidence},
                {"class": "Basal Cell Carcinoma", "confidence": (100 - percentage) / 200},
                {"class": "Squamous Cell Carcinoma", "confidence": (100 - percentage) / 300}
            ]
        elif percentage > 50:
            prediction = "Basal Cell Carcinoma"
            confidence = percentage / 100
            other_predictions = [
                {"class": "Basal Cell Carcinoma", "confidence": confidence},
                {"class": "Melanoma", "confidence": (100 - percentage) / 200},
                {"class": "Actinic Keratosis", "confidence": (100 - percentage) / 300}
            ]
        elif percentage > 30:
            prediction = "Actinic Keratosis"
            confidence = percentage / 100
            other_predictions = [
                {"class": "Actinic Keratosis", "confidence": confidence},
                {"class": "Seborrheic Keratosis", "confidence": (100 - percentage) / 150},
                {"class": "Nevus", "confidence": (100 - percentage) / 250}
            ]
        else:
            prediction = "Nevus (Benign Mole)"
            confidence = (100 - percentage) / 100
            other_predictions = [
                {"class": "Nevus (Benign Mole)", "confidence": confidence},
                {"class": "Seborrheic Keratosis", "confidence": percentage / 150},
                {"class": "Dermatofibroma", "confidence": percentage / 200}
            ]
        
        return jsonify({
            "success": True,
            "predicted_class": prediction,
            "confidence": confidence,
            "top_predictions": other_predictions
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)