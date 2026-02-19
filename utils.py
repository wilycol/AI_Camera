import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from db_config import faces_collection, logs_collection
from datetime import datetime
from PIL import Image

# Initialize models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# margin=20 for better face capture, landmarks=True for future alignment
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20, device=device, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face(img):
    """
    Detect multiple faces and return list of face tensors.
    Returns: (list of face_tensors, list of boxes, list of probs)
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Detect all faces
    # boxes: [num_faces, 4], probs: [num_faces]
    boxes, probs = mtcnn.detect(img_pil)
    
    if boxes is None:
        return [], [], []

    face_tensors = []
    valid_boxes = []
    valid_probs = []

    for i, prob in enumerate(probs):
        if prob > 0.90:  # Confidence threshold for detection
            # Crop and resize face
            face = mtcnn.extract(img_pil, boxes[i:i+1], save_path=None)
            face_tensors.append(face.squeeze(0)) # Move to (3, 160, 160)
            valid_boxes.append(boxes[i])
            valid_probs.append(prob)

    return face_tensors, valid_boxes, valid_probs

def forward_pass(face_tensor):
    """
    Generate L2-Normalized 512-d embedding for a face.
    """
    with torch.no_grad():
        # face_tensor is (3, 160, 160)
        # InceptionResnetV1 expects [batch, 3, 160, 160]
        embedding = resnet(face_tensor.unsqueeze(0).to(device))
        # L2 Normalization
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        
    return embedding.squeeze().cpu().numpy().astype("float32")

def save_embedding(label, embedding, role="user"):
    """
    Save normalized embedding and metadata to MongoDB (Scalable Schema).
    """
    faces_collection.update_one(
        {"label": label},
        {
            "$set": {
                "embedding": embedding.tolist(),
                "role": role,
                "active": True,
                "updated_at": datetime.now()
            },
            "$setOnInsert": {
                "created_at": datetime.now()
            }
        },
        upsert=True
    )

def load_embeddings():
    """Load all active known embeddings from MongoDB"""
    known_faces = list(faces_collection.find({"active": True}, {"_id": 0}))
    for face in known_faces:
        face["embedding"] = np.array(face["embedding"])
    return known_faces

def identify_face(embedding, known_faces, threshold=0.75):
    """
    Identify face using Cosine Similarity (Dot Product if normalized).
    """
    if not known_faces:
        return "Unknown", 0.0
        
    best_score = -1.0
    best_label = "Unknown"
    
    for face in known_faces:
        stored = face["embedding"]
        # Since both are L2-normalized, Dot Product == Cosine Similarity
        score = np.dot(embedding, stored)
        
        if score > best_score:
            best_score = score
            best_label = face["label"]
            
    if best_score < threshold:
        return "Unknown", float(best_score)
        
    return best_label, float(best_score)

def log_event(label, score, authorized=True):
    """Log identification event (Audit Ready)"""
    logs_collection.insert_one({
        "label": label,
        "score": float(score),
        "authorized": authorized,
        "timestamp": datetime.now()
    })

def allowed_file(filename):
    """Check if the file has an allowed image extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
