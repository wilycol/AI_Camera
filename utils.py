import cv2
import numpy as np
from db_config import faces_collection, logs_collection
from datetime import datetime

def load_model(model_path):
    """Load the model from the given path (Keras .h5 format)"""
    from keras.models import load_model as keras_load
    return keras_load(model_path)

def get_face(img, detector):
    """Detect and crop face from the image using MTCNN"""
    results = detector.detect_faces(img)
    if not results:
        return None, None
    
    # Extract the first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    
    # Resize to FaceNet input size (160x160)
    face = cv2.resize(face, (160, 160))
    return face, (x1, y1, width, height)

def forward_pass(model, face_pixels):
    """Generate 128-d embedding for a face using FaceNet"""
    # Preprocess
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    
    # Predict
    yhat = model.predict(samples)
    return yhat[0]

def save_embedding(label, embedding):
    """Save face label and embedding to MongoDB"""
    faces_collection.update_one(
        {"label": label},
        {"$set": {"embedding": embedding.tolist()}},
        upsert=True
    )

def load_embeddings():
    """Load all known embeddings from MongoDB"""
    known_faces = list(faces_collection.find({}, {"_id": 0}))
    for face in known_faces:
        face["embedding"] = np.array(face["embedding"])
    return known_faces

def identify_face(embedding, known_faces, threshold=10.0):
    """Identify a face by comparing embeddings using Euclidean distance"""
    min_dist = float('inf')
    found_label = "Unknown"
    
    for face in known_faces:
        dist = np.linalg.norm(embedding - face["embedding"])
        if dist < min_dist:
            min_dist = dist
            found_label = face["label"]
            
    if min_dist > threshold:
        return "Unknown", min_dist
    return found_label, min_dist

def log_event(label, confidence):
    """Log a recognition event to MongoDB"""
    logs_collection.insert_one({
        "label": label,
        "confidence": float(confidence),
        "timestamp": datetime.now()
    })

def allowed_file(filename):
    """Check if the file has an allowed image extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_file_extension(filename):
    """Remove the file extension from a filename"""
    return filename.rsplit('.', 1)[0]

def save_image(image, path):
    """Save image to specified path"""
    cv2.imwrite(path, image)
