import os
import cv2
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from face_rec import recognize_faces, BASE_GALLERY_DIR

# FastAPI app setup
app = FastAPI(
    title="Face Recognition API",
    description="Simple API for face recognition"
)


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "Face Recognition API is running"}


@app.post("/recognize")
async def recognize_image(
    image: UploadFile = File(...),
    gallery_name: str = "default"
):
    """Recognize faces in an uploaded image"""
    
    # Read and decode image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Build gallery path
    gallery_path = os.path.join(BASE_GALLERY_DIR, f"{gallery_name}.pth")
    
    if not os.path.exists(gallery_path):
        raise HTTPException(status_code=400, detail=f"Gallery '{gallery_name}' not found")

    # Perform face recognition
    result_img, faces = recognize_faces(img, gallery_paths=[gallery_path])
    
    # Save result image
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/result.jpg", result_img)

    # Prepare response
    recognized_faces = []
    for face in faces:
        recognized_faces.append({
            "identity": face["identity"],
            "confidence": float(face["similarity"]),
            "bounding_box": [int(x) for x in face["bounding_box"]]
        })

    return JSONResponse(content={
        "success": True,
        "faces_detected": len(recognized_faces),
        "faces": recognized_faces
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)