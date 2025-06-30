import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Union, Any
from ultralytics import YOLO
import torch
from scipy.spatial.distance import cosine
from PIL import Image
from torchvision import transforms
from gallery_manager import load_model

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "LightCNN_29Layers_V2_checkpoint.pth.tar")
DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "yolo", "weights", "yolo11n-face.pt")
BASE_GALLERY_DIR = os.path.join(BASE_DIR, "gallery", "galleries")

# Create necessary directories
os.makedirs(BASE_GALLERY_DIR, exist_ok=True)


def recognize_faces(
    frame: np.ndarray, 
    gallery_paths: Union[str, List[str]], 
    model_path: str = DEFAULT_MODEL_PATH,
    yolo_path: str = DEFAULT_YOLO_PATH,
    threshold: float = 0.50,
    model=None,
    device=None,
    yolo_model=None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Recognize faces in a given frame using one or more galleries.
    
    Args:
        frame: Input image (numpy array in BGR format from cv2)
        gallery_paths: Single gallery path or list of gallery paths
        model_path: Path to LightCNN model
        yolo_path: Path to YOLO face detection model
        threshold: Minimum similarity threshold (0-1)
        model: Pre-loaded model (optional)
        device: Pre-loaded device (optional)
        yolo_model: Pre-loaded YOLO model (optional)
        
    Returns:
        Tuple containing:
            - Annotated frame with bounding boxes and labels
            - List of recognized identities with details
    """
    if isinstance(gallery_paths, str):
        gallery_paths = [gallery_paths]
    
    # Load models if not provided
    if model is None or device is None:
        model, device = load_model(model_path)
    
    if yolo_model is None:
        yolo_model = YOLO(yolo_path)
    
    # Load and combine galleries
    combined_gallery = {}
    for gallery_path in gallery_paths:
        if os.path.exists(gallery_path):
            try:
                gallery_data = torch.load(gallery_path)
                if isinstance(gallery_data, dict):
                    if "identities" in gallery_data:
                        combined_gallery.update(gallery_data["identities"])
                    else:
                        combined_gallery.update(gallery_data)
            except Exception as e:
                print(f"Error loading gallery {gallery_path}: {e}")
    
    if not combined_gallery:
        return frame, []
    
    # Face detection
    face_detections = []
    results = yolo_model(frame, conf=0.5)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Add padding around face
            h, w = frame.shape[:2]
            face_w, face_h = x2 - x1, y2 - y1
            pad_x = int(face_w * 0.2)
            pad_y = int(face_h * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            # Skip faces that are too small
            if (x2 - x1) < 32 or (y2 - y1) < 32:
                continue
        
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                continue
                
            # Convert to PIL and prepare for model
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            
            face_tensor_transformed = transform(face_pil)
            face_tensor = face_tensor_transformed.unsqueeze(0).to(device)
            
            # Extract embedding
            with torch.no_grad():
                _, embedding = model(face_tensor)
                face_embedding = embedding.cpu().squeeze().numpy()
            
            # Find best match
            best_match = None
            best_similarity = threshold
            
            for identity, gallery_embedding in combined_gallery.items():
                similarity = 1 - cosine(face_embedding, gallery_embedding)
                if similarity > best_similarity:
                    best_match = identity
                    best_similarity = similarity
            
            # Store detection
            detection = {
                "bounding_box": [x1, y1, x2, y2],
                "identity": best_match if best_match else "Unknown",
                "similarity": best_similarity if best_match else 0.0
            }
            face_detections.append(detection)
    
    # Annotate frame
    annotated_frame = frame.copy()
    for detection in face_detections:
        x1, y1, x2, y2 = detection["bounding_box"]
        identity = detection["identity"]
        similarity = detection["similarity"]
        
        # Draw bounding box
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{identity}: {similarity:.2f}" if identity != "Unknown" else "Unknown"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return annotated_frame, face_detections
