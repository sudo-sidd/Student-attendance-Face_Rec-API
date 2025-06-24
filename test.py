import os
import cv2
import numpy as np
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from face_rec import recognize_faces, BASE_GALLERY_DIR
import base64
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from io import BytesIO

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# FastAPI app setup
app = FastAPI(
    title="Face Recognition API",
    description="API for recognizing faces in uploaded images"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recognize", summary="Recognize faces and return both the annotated image and detection results")
async def recognize_image(
    image: UploadFile = File(...),
    dept_id: List[str] = Form(...),
    year: List[str] = Form(...),
    section_students: List[str] = Form(...),
):
    """Recognize faces in an uploaded image and return both the base64-encoded image and detection results"""
    if len(dept_id) != len(year):
        raise HTTPException(status_code=400, detail="dept_id and year lists must be of equal length")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Example: build gallery paths based on dept_id and year
    gallery_paths = []
    for d, y in zip(dept_id, year):
        gallery_name = f"{d}_{y}.pth"
        gallery_path = os.path.join(BASE_GALLERY_DIR, gallery_name)
        if os.path.exists(gallery_path): 
            gallery_paths.append(gallery_path)

    if not gallery_paths:
        raise HTTPException(status_code=400, detail="No valid galleries found")

    result_img, faces = recognize_faces(img, gallery_paths=gallery_paths)
    cv2.imwrite("outputs/result.jpg", result_img)
    # Encode the result image to base64
    _, buffer = cv2.imencode('.jpg', result_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    # Prepare face recognition results
    serializable_faces = []
    for face in faces:
        serializable_face = {
            "identity": face["identity"],
            "similarity": float(face["similarity"]),
            "bounding_box": [int(x) for x in face["bounding_box"]]
        }
        serializable_faces.append(serializable_face)
    
    # Return both the image and results in a single JSON response
    return JSONResponse(content={
        "count": len(serializable_faces),
        "image_base64": base64_img,
        "faces": serializable_faces,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5564)