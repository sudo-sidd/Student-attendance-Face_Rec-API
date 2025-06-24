import os
import json
import uuid
import re
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
from pathlib import Path
import logging
from typing import List, Optional
from datetime import datetime, time, timedelta
import pandas as pd
import io
import torch
import cv2
import numpy as np
import base64
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from backend.LightCNN.light_cnn import LightCNN_29Layers_v2  
from dotenv import load_dotenv
import uvicorn
import bcrypt
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Student Attendance API", docs_url=None, redoc_url=None)

# CORS setup - Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_gallery(dept_name: str, year: int):
    try:
        gallery_path = f"./gallery/gallery_{dept_name}_{year}.pth"
        if not Path(gallery_path).exists():
            logger.warning(f"Gallery file {gallery_path} not found")
            return {}
        gallery_data = torch.load(gallery_path, map_location='cpu')
        # Handle np.ndarray or tensor values
        gallery = {}
        for k, v in gallery_data.items():
            if isinstance(v, np.ndarray):
                gallery[k] = v
            elif isinstance(v, torch.Tensor):
                gallery[k] = v.cpu().numpy()
            else:
                logger.warning(f"Skipping invalid embedding for {k}: {type(v)}")
        logger.info(f"Loaded gallery {gallery_path} with {len(gallery)} identities")
        return gallery
    except Exception as e:
        logger.error(f"Error loading gallery {gallery_path}: {e}")
        return {}

def process_image(image_bytes, threshold=0.45, gallery=None, save_path=None, filename_base=None, img_index=None, section_students=None):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        # Create image directory if it doesn't exist
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        # Save original image if path is provided
        if save_path and filename_base:
            suffix = f"-{img_index}" if img_index is not None else ""
            orig_filename = f"{filename_base}{suffix}_original.jpg"
            orig_path = os.path.join(save_path, orig_filename)
            cv2.imwrite(orig_path, img)
            logger.info(f"Saved original image to {orig_path}")

        result_img = img.copy()
        detected_ids = set()
                
        # Detect faces using YOLO
        results = yolo_model(img)
        logger.info(f"YOLO detected {len(results[0].boxes)} faces")

        # Step 1: Get all faces and their embeddings
        faces_data = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = img.shape[:2]
                face_w, face_h = x2 - x1, y2 - y1
                pad_x, pad_y = int(face_w * 0.2), int(face_h * 0.2)
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    print(f"Skipping face too small ({x2-x1}x{y2-y1})")
                    continue

                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    logger.warning(f"Empty face crop at {x1},{y1},{x2},{y2}")
                    continue

                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_pil = Image.fromarray(gray_face)
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    _, embedding = face_model(face_tensor)
                    face_embedding = embedding.cpu().squeeze().numpy()
                
                # Store all potential matches for this face
                matches = []
                for identity, gallery_embedding in gallery.items():
                    # Only consider matches for students in this section if section_id is provided
                    if section_students and identity not in section_students:
                        continue
                        
                    similarity = 1 - cosine(face_embedding, gallery_embedding)
                    if similarity > threshold:
                        matches.append((identity, similarity))
                
                # Sort matches by similarity (highest first)
                matches.sort(key=lambda x: x[1], reverse=True)
                
                # Store all face data
                faces_data.append({
                    'coords': (x1, y1, x2, y2),
                    'embedding': face_embedding,
                    'matches': matches,
                    'best_match': "Unknown",
                    'best_score': -1
                })
        
        # Step 2: Assign identities based on highest confidence without duplicates
        used_identities = set()
        
        # First pass: assign identities to faces with highest confidence
        for face in sorted(faces_data, key=lambda x: max([m[1] for m in x['matches']]) if x['matches'] else 0, reverse=True):
            for identity, score in face['matches']:
                if identity not in used_identities:
                    face['best_match'] = identity
                    face['best_score'] = score
                    used_identities.add(identity)
                    detected_ids.add(identity)
                    break
        
        # Step 3: Draw the results
        for face in faces_data:
            x1, y1, x2, y2 = face['coords']
            best_match = face['best_match']
            best_score = face['best_score']
            
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{best_match} ({best_score:.2f})" if best_match != "Unknown" else "Unknown"
            cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the annotated image if path is provided
        if save_path and filename_base:
            suffix = f"-{img_index}" if img_index is not None else ""
            annotated_filename = f"{filename_base}{suffix}_annotated.jpg"
            annotated_path = os.path.join(save_path, annotated_filename)
            cv2.imwrite(annotated_path, result_img)
            logger.info(f"Saved annotated image to {annotated_path}")

        _, buffer = cv2.imencode('.jpg', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64, detected_ids
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None, []

def generate_image_filename(dept_name, year, date, suffix=""):
    # Clean inputs to make them safe for filenames
    dept_clean = re.sub(r'[^\w]', '_', dept_name)
    date_clean = re.sub(r'[^\w-]', '_', date)
    
    # Create base filename
    base_name = f"{dept_clean}_{year}_{date_clean}"
    
    # Add suffix if provided
    if suffix:
        base_name = f"{base_name}_{suffix}"
        
    # Add unique identifier to prevent collisions
    unique_id = uuid.uuid4().hex[:8]
    
    return f"{base_name}_{unique_id}"

# Startup event
@app.on_event("startup")
async def startup_event():
    
    global face_model, yolo_model, device

    model_path = "backend/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"
    yolo_path = "backend/yolo/weights/yolo11n-face.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} for inference")

    try:
        face_model = LightCNN_29Layers_v2(num_classes=100)
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.get("state_dict", checkpoint).items() if 'fc2' not in k}
        face_model.load_state_dict(new_state_dict, strict=False)
        face_model = face_model.to(device)
        face_model.eval()
        logger.info("Face recognition model loaded")
    except Exception as e:
        logger.error(f"Error loading face model: {e}")
        raise RuntimeError(f"Failed to load face recognition model: {e}")

    try:
        yolo_model = YOLO(yolo_path)
        logger.info("YOLO face detection model loaded")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise RuntimeError(f"Failed to load YOLO model: {e}")

@app.post("/process-images")
async def process_images(
    images: List[UploadFile] = File(...),
    dept_name: str = Form(...),
    year: int = Form(...),
    student_reg: List[int] = Form(...),
):
    threshold = 0.450
    if face_model is None or yolo_model is None:
        logger.error("Models not loaded")
        raise HTTPException(status_code=500, detail="Models not loaded")

    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    try:
        # Load gallery
        gallery = load_gallery(dept_name, year)
        if not gallery:
            logger.warning("Empty gallery; proceeding with all students marked absent")
            
            return {
                "message": "No gallery data available; all students marked absent"
            }

        if not student_reg:
            logger.warning(f"No students found for section {dept_name} year {year}")
            return {
                "message": "No students data passed"
            }

        # Process images
        detected_students = set()
        images_base64 = []
        save_path = f"./processed_images/{dept_name}/{year}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        filename_base = generate_image_filename(dept_name, year, datetime.now())
        for img_index, image in enumerate(images):
            contents = await image.read()
            img_base64, detected_ids = process_image(contents, threshold, gallery, save_path, filename_base, img_index, student_reg)
            images_base64.append(img_base64)
            detected_students.update(detected_ids)
            
        # Prepare attendance
        attendance = [
            {
                "register_number": reg_num,
                "is_present": 1 if reg_num in detected_students else 0
            }
            for reg_num in student_reg
        ]

        logger.info(f"Processed {len(images)} images, detected {len(detected_students)} students")
        return {
            "attendance": attendance,
            "images_base64": images_base64
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in /process-images: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
