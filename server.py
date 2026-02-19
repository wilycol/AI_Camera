from flask import Flask, request, jsonify
import numpy as np
import cv2
import utils
import os

app = Flask(__name__)

@app.route('/register', methods=['POST'])
def register():
    """Register a new face embedding with optional role (Scalable Phase 1)"""
    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Missing file or label'}), 400
    
    file = request.files['file']
    label = request.form['label']
    role = request.form.get('role', 'user') # Default role
    
    if file and utils.allowed_file(file.filename):
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        # get_face returns lists now
        faces, boxes, probs = utils.get_face(img)
        
        if not faces:
            return jsonify({'error': 'No face detected or low confidence'}), 400
            
        # Register the most confident face
        try:
            embedding = utils.forward_pass(faces[0])
            utils.save_embedding(label, embedding, role=role)
            return jsonify({
                'message': f'Face registered for {label} with role {role}',
                'probe_confidence': float(probs[0])
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/predictImage', methods=['POST'])
def predict_image():
    """Detect and recognize faces (Prepared for Multi-face)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file and utils.allowed_file(file.filename):
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        # get_face returns lists
        faces, boxes, probs = utils.get_face(img)
        
        if not faces:
            return jsonify({'error': 'No face found'}), 400
            
        # In Phase 1, we process the first face detected
        # In Phase 2, this will be a loop over faces
        results = []
        known_faces = utils.load_embeddings()
        
        try:
            # Process primary face
            embedding = utils.forward_pass(faces[0])
            label, score = utils.identify_face(embedding, known_faces)
            
            authorized = (label != "Unknown")
            utils.log_event(label, score, authorized=authorized)
            
            results.append({
                'label': label,
                'score': float(score),
                'box': boxes[0].tolist(),
                'authorized': authorized
            })
            
            return jsonify({
                'status': 'success',
                'phase': 1,
                'results': results,
                'message': 'Face recognized' if authorized else 'Unauthorized Access'
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'backend': 'pytorch-scalable-base',
        'device': str(utils.device),
        'phase': 1
    }), 200

if __name__ == '__main__':
    print("--------------------------------------------------")
    print(f"Starting AI_Camera (Scalable Base Phase 1)")
    print(f"Backend: PyTorch | Device: {utils.device}")
    print("--------------------------------------------------")
    app.run(host='0.0.0.0', port=5000)