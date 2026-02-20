import cv2
import torch
import numpy as np
import utils
from PIL import Image
import time

def main():
    print("--- Starting AI_Camera Live Demo ---")
    print(f"Using Device: {utils.device}")
    
    # Load known faces from DB (Local or Atlas)
    print("Loading known faces...")
    known_faces = utils.load_embeddings()
    print(f"Loaded {len(known_faces)} faces.")

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit, 'r' to register current face.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detection and identification
        # We use utils.get_face to detect all faces
        faces, boxes, probs = utils.get_face(frame)

        if faces:
            for i, face_tensor in enumerate(faces):
                box = boxes[i].astype(int)
                prob = probs[i]

                # Draw Bounding Box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # Vectorization & Identification
                # In a real app, we might do this only every few frames to save CPU
                start_time = time.time()
                embedding = utils.forward_pass(face_tensor)
                label, score = utils.identify_face(embedding, known_faces)
                processing_time = (time.time() - start_time) * 1000

                # Display Info
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                text = f"{label} ({score:.2f})"
                cv2.putText(frame, text, (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Show "Vectorizing..." info
                cv2.putText(frame, f"Vector: 512-d | {processing_time:.1f}ms", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show frame
        cv2.imshow('AI_Camera - Scalable Base Phase 1', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Simple registration helper for the demo
            if faces:
                name = input("Enter name for this face: ")
                role = input("Enter role (default: user): ") or "user"
                embedding = utils.forward_pass(faces[0])
                utils.save_embedding(name, embedding, role=role)
                print(f"Registered {name}!")
                # Refresh known faces
                known_faces = utils.load_embeddings()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
