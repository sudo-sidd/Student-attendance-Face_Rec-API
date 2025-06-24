import os
import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple, Union, Any
from ultralytics import YOLO
import torch
from scipy.spatial.distance import cosine
from PIL import Image
from torchvision import transforms
import random
import albumentations as A
from gallery_manager import create_gallery, update_gallery, load_model, extract_embedding

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR,"checkpoints", "LightCNN_29Layers_V2_checkpoint.pth.tar")
DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "yolo", "weights", "yolo11n-face.pt")
BASE_DATA_DIR = os.path.join(BASE_DIR, "gallery", "data")
BASE_GALLERY_DIR = os.path.join(BASE_DIR, "gallery", "galleries")

# Create necessary directories
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(BASE_GALLERY_DIR, exist_ok=True)

def get_gallery_path(year: str, department: str) -> str:
    """Generate a standardized gallery path based on batch year and department"""
    filename = f"gallery_{department}_{year}.pth"
    return os.path.join(BASE_GALLERY_DIR, filename)

def get_data_path(year: str, department: str) -> str:
    """Generate a standardized data path for storing preprocessed faces"""
    return os.path.join(BASE_DATA_DIR, f"{department}_{year}")

def extract_frames(video_path: str, output_dir: str, max_frames: int = 30, interval: int = 10) -> List[str]:
    """
    Extract frames from a video at specified intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract
        interval: Extract a frame every 'interval' frames
    
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return frame_paths

def detect_and_crop_faces(image_path: str, output_dir: str, yolo_path: str = DEFAULT_YOLO_PATH) -> List[str]:
    """
    Detect, crop, and preprocess faces from an image using YOLO
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save preprocessed face images
        yolo_path: Path to YOLO model weights
        
    Returns:
        List of paths to preprocessed face images
    """
    print(f"Processing image: {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(yolo_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    
    results = model(img)
    print(f"YOLO detected {sum(len(r.boxes) for r in results)} faces in {image_path}")
    
    face_paths = []
    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            h, w = img.shape[:2]
            face_w = x2 - x1
            face_h = y2 - y1
            pad_x = int(face_w * 0.2)
            pad_y = int(face_h * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            print(f"Face {j} dimensions before padding: {x2-x1}x{y2-y1}")
            print(f"Face {j} dimensions after padding: {max(0, x1-pad_x)}-{min(w, x2+pad_x)}x{max(0, y1-pad_y)}-{min(h, y2+pad_y)}")
            
            if (x2 - x1) < 32 or (y2 - y1) < 32:
                print(f"Skipping face {j} in {image_path} - too small ({x2-x1}x{y2-y1})")
                continue
                
            face = img[y1:y2, x1:x2]
            
            if face.size == 0 or face.shape[0] <= 0 or face.shape[1] <= 0:
                print(f"Skipping face {j} in {image_path} - invalid dimensions")
                continue
            
            if len(face.shape) == 3:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            else:
                gray = face
                
            resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            normalized = resized.astype(np.float32) / 255.0
            equalized = cv2.equalizeHist(resized)
            
            face_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_face_{j}.jpg")
            cv2.imwrite(face_path, equalized)
            face_paths.append(face_path)
    
    return face_paths

def create_face_augmentations():
    """Create a set of specific augmentations for face images"""
    augmentations = [
        A.Compose([
            A.Resize(height=32, width=32),
            A.Resize(height=128, width=128)
        ]),
        A.Compose([
            A.Resize(height=24, width=24),
            A.Resize(height=128, width=128)
        ]),
        A.RandomBrightnessContrast(p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        A.GaussianBlur(p=1.0, blur_limit=(3, 7)),
        A.Compose([
            A.Resize(height=48, width=48),
            A.Resize(height=128, width=128),
            A.GaussianBlur(p=1.0, blur_limit=(2, 5))
        ]),
        A.Compose([
            A.Resize(height=32, width=32),
            A.Resize(height=128, width=128),
            A.GaussianBlur(p=1.0, blur_limit=(2, 5))
        ]),
    ]
    return augmentations

def augment_face_image(image, num_augmentations=2):
    """
    Generate augmented versions of a face image in-memory
    
    Args:
        image: Original face image (numpy array)
        num_augmentations: Number of augmented versions to generate
    
    Returns:
        List of augmented images (numpy arrays)
    """
    augmentations_list = create_face_augmentations()
    augmented_images = []
    
    for i in range(num_augmentations):
        selected_aug = random.choice(augmentations_list)
        if isinstance(selected_aug, A.Compose):
            augmented = selected_aug(image=image)
        else:
            aug_pipeline = A.Compose([selected_aug])
            augmented = aug_pipeline(image=image)
        augmented_images.append(augmented['image'])
    
    return augmented_images

def process_videos_directory(videos_dir: str, year: str, department: str) -> Dict[str, Any]:
    """
    Process all videos in a directory, extract frames, detect faces,
    and update or create a gallery file
    
    Args:
        videos_dir: Path to directory containing videos
        year: Batch year
        department: Department name
    
    Returns:
        Dictionary containing processing statistics
    """
    data_path = get_data_path(year, department)
    gallery_path = get_gallery_path(year, department)
    os.makedirs(data_path, exist_ok=True)
    
    processed_videos = 0
    processed_frames = 0
    extracted_faces = 0
    failed_videos = []
    
    model = load_model(DEFAULT_MODEL_PATH)
    student_embeddings = {}
    
    for filename in os.listdir(videos_dir):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
            
        student_name = os.path.splitext(filename)[0]
        video_path = os.path.join(videos_dir, filename)
        
        frames_dir = os.path.join(data_path, student_name, "frames")
        faces_dir = os.path.join(data_path, student_name, "faces")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(faces_dir, exist_ok=True)
        
        try:
            frame_paths = extract_frames(video_path, frames_dir)
            processed_frames += len(frame_paths)
            
            student_faces = []
            for frame_path in frame_paths:
                face_paths = detect_and_crop_faces(frame_path, faces_dir)
                extracted_faces += len(face_paths)
                student_faces.extend(face_paths)
            
            if student_faces:
                embeddings = [extract_embedding(face_path, model) for face_path in student_faces]
                student_embeddings[student_name] = np.mean(embeddings, axis=0)
            
            processed_videos += 1
            
        except Exception as e:
            print(f"Error processing video {filename}: {e}")
            failed_videos.append(filename)
    
    gallery_updated = False
    if student_embeddings:
        if os.path.exists(gallery_path):
            update_gallery(gallery_path, student_embeddings)
        else:
            create_gallery(gallery_path, student_embeddings)
        gallery_updated = True
    
    return {
        "processed_videos": processed_videos,
        "processed_frames": processed_frames,
        "extracted_faces": extracted_faces,
        "failed_videos": failed_videos,
        "gallery_updated": gallery_updated,
        "gallery_path": gallery_path
    }

def get_gallery_info(gallery_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a gallery file
    
    Args:
        gallery_path: Path to gallery file
    
    Returns:
        Dictionary with gallery info or None if file doesn't exist
    """
    if not os.path.exists(gallery_path):
        return None
    
    try:
        gallery_data = torch.load(gallery_path)
        if isinstance(gallery_data, dict) and "identities" in gallery_data:
            identities = gallery_data["identities"]
        else:
            identities = list(gallery_data.keys())
            
        return {
            "gallery_path": gallery_path,
            "identities": identities,
            "count": len(identities)
        }
    except Exception as e:
        print(f"Error loading gallery file: {e}")
        return None

def recognize_faces(
    frame: np.ndarray, 
    gallery_paths: Union[str, List[str]], 
    model_path: str = DEFAULT_MODEL_PATH,
    yolo_path: str = DEFAULT_YOLO_PATH,
    threshold: float = 0.45,
    model=None,
    device=None,
    yolo_model=None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Recognize faces in a given frame using one or more galleries.
    Implements a no-duplicate rule where each identity appears only once.
    
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
    
    if model is None or device is None:
        model, device = load_model(model_path)
    
    if yolo_model is None:
        yolo_model = YOLO(yolo_path)
    
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
    
    face_detections = []
    results = yolo_model(frame, conf=0.5)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            h, w = frame.shape[:2]
            face_w, face_h = x2 - x1, y2 - y1
            pad_x = int(face_w * 0.2)
            pad_y = int(face_h * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            if (x2 - x1) < 32 or (y2 - y1) < 32:
                print(f"Skipping face - too small ({x2-x1}x{y2-y1})")
                continue
        
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                continue
                
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, embedding = model(face_tensor)
                face_embedding = embedding.cpu().squeeze().numpy()
            
            matches = []
            for identity, gallery_embedding in combined_gallery.items():
                similarity = 1 - cosine(face_embedding, gallery_embedding)
                if similarity >= threshold:
                    matches.append((identity, similarity))
            
            matches.sort(key=lambda x: x[1], reverse=True)
            
            face_detections.append({
                "bbox": (x1, y1, x2, y2),
                "matches": matches,
                "embedding": face_embedding
            })
    
    face_detections.sort(key=lambda x: x["matches"][0][1] if x["matches"] else 0, reverse=True)
    
    assigned_identities = set()
    detected_faces = []
    
    for face in face_detections:
        x1, y1, x2, y2 = face["bbox"]
        matches = face["matches"]
        
        best_match = None
        best_score = 0.0
        
        for identity, score in matches:
            if identity not in assigned_identities:
                best_match = identity
                best_score = float(score)
                break
        
        if best_match:
            detected_faces.append({
                "identity": best_match,
                "similarity": best_score,
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
            })
            assigned_identities.add(best_match)
        else:
            detected_faces.append({
                "identity": "Unknown",
                "similarity": 0.0,
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
            })
    
    result_img = frame.copy()
    
    for face_info in detected_faces:
        identity = face_info["identity"]
        similarity = face_info["similarity"]
        x1, y1, x2, y2 = face_info["bounding_box"]
        
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{identity} ({similarity:.2f})" if identity != "Unknown" else "Unknown"
        
        text_bg_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_w, text_h = text_size
        
        cv2.rectangle(result_img, 
                     (x1, y1 - text_h - 8), 
                     (x1 + text_w, y1), 
                     text_bg_color, -1)
        
        cv2.putText(result_img, 
                   label, 
                   (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2)
    
    return result_img, detected_faces


