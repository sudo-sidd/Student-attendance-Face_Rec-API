# Face Recognition API

A simple face recognition API built with FastAPI and LightCNN for accurate face detection and recognition.

## Features

- **Face Detection**: YOLO-based face detection
- **Face Recognition**: LightCNN feature extraction for face recognition
- **RESTful API**: Simple FastAPI web service
- **Gallery System**: Load pre-built face galleries for recognition

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Checkpoints

download checkpoints file from google drive:
```bash
https://drive.google.com/file/d/1CUdlD83CYpiC-KedxLM9b8nG0ZAltsP4/view?usp=sharing
```

```bash
mkdir -p checkpoints
# Download LightCNN checkpoint to checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar
# The YOLO model will be downloaded automatically on first use
```

### 3. Prepare Face Gallery

Create a face gallery from a directory of face images:

```bash
python gallery_manager.py --model_path checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar \
                         --data_dir /path/to/face/images \
                         --output_path gallery/galleries/my_gallery.pth
```

Directory structure for face images:
```
face_images/
├── person1/
│   ├── face1.jpg
│   └── face2.jpg
├── person2/
│   └── face1.jpg
└── person3/
    ├── face1.jpg
    ├── face2.jpg
    └── face3.jpg
```

### 4. Start the API Server

```bash
python api.py
```

The API will be available at: http://localhost:8000

## API Usage

### Recognize Faces

**POST** `/recognize`

- **image**: Image file (multipart/form-data)
- **gallery_name**: Name of the gallery to use (default: "default")

**Example:**
```bash
curl -X POST "http://localhost:8000/recognize" \
     -F "image=@test_image.jpg" \
     -F "gallery_name=my_gallery"
```

**Response:**
```json
{
  "success": true,
  "faces_detected": 2,
  "faces": [
    {
      "identity": "person1",
      "confidence": 0.87,
      "bounding_box": [100, 150, 200, 250]
    },
    {
      "identity": "Unknown",
      "confidence": 0.0,
      "bounding_box": [300, 100, 400, 200]
    }
  ]
}
```

### Health Check

**GET** `/`

Returns API status.

## Testing

Run the test script:

```bash
python test.py
```

Make sure to place a test image named `test_image.jpg` in the project directory.

## Project Structure

```
face_final/
├── api.py                  # FastAPI web service
├── face_rec.py            # Core face recognition module
├── gallery_manager.py     # Gallery creation utilities
├── test.py               # Test script
├── requirements.txt      # Dependencies
├── checkpoints/          # Model checkpoints
├── gallery/
│   └── galleries/        # Face galleries (.pth files)
├── outputs/              # Output images
└── yolo/
    └── weights/          # YOLO model weights
```

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **OpenCV**: Computer vision
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation
- **Pillow**: Image processing
- **SciPy**: Scientific computing
- **NumPy**: Numerical computing

## License

MIT License
