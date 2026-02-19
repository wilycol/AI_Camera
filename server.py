from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
from mtcnn import MTCNN
import utils

app = Flask(__name__)

# Initialize detector
detector = MTCNN()

# Model Path (User needs to download this)
MODEL_PATH = 'facenet_keras.h5'
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = utils.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please download it first.")
    return model

@app.route('/register', methods=['POST'])
def register():
    """Register a new face embedding into the database"""
    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Missing file or label'}), 400
    
    file = request.files['file']
    label = request.form['label']
    
    if file and utils.allowed_file(file.filename):
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        face, _ = utils.get_face(img, detector)
        if face is None:
            return jsonify({'error': 'No face detected'}), 400
            
        try:
            embedding = utils.forward_pass(get_model(), face)
            utils.save_embedding(label, embedding)
            return jsonify({'message': f'Face registered for {label}'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/predictImage', methods=['POST'])
def predict_image():
    """Detect and recognize faces in an uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file and utils.allowed_file(file.filename):
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        face, _ = utils.get_face(img, detector)
        if face is None:
            return jsonify({'error': 'No face found'}), 400
            
        try:
            # Generate embedding
            embedding = utils.forward_pass(get_model(), face)
            
            # Identify face
            known_faces = utils.load_embeddings()
            label, distance = utils.identify_face(embedding, known_faces)
            
            # Log event
            utils.log_event(label, distance)
            
            return jsonify({
                'result': label,
                'distance': float(distance),
                'status': 'Face recognized' if label != "Unknown" else "Stranger detected"
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    # We do not run the server automatically as per user request
    app.run(host='0.0.0.0', port=5000)