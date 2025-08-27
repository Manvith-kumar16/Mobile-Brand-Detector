import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# --- IMPORTANT: MAKE SURE THESE FILE PATHS ARE CORRECT ---
MODEL_PATH = 'mobile_detector_model.h5'
LABELS_PATH = 'class_labels.txt'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]
    print("Model and labels loaded successfully.")
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    class_labels = []

def preprocess_image(img_file):
    img = Image.open(io.BytesIO(img_file.read()))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            processed_image = preprocess_image(file)
            predictions = model.predict(processed_image)
            predicted_index = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_index]

            parts = predicted_label.split('_')
            brand = parts[0]
            model_name = " ".join(parts[1:])

            return jsonify({
                'brand': brand,
                'model': model_name
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
