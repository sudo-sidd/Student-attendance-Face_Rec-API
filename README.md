# Face Recognition System

A comprehensive face recognition system built with FastAPI, featuring face detection, recognition, and gallery management capabilities. The system uses YOLO for face detection and LightCNN for feature extraction to provide accurate and efficient face recognition.

## Features

- **Face Detection**: YOLO-based face detection with high accuracy
- **Face Recognition**: Deep learning-based face recognition using LightCNN
- **Gallery Management**: Create and manage multiple face galleries
- **RESTful API**: FastAPI-based web service with interactive documentation
- **Real-time Processing**: Efficient face recognition with optimized models
- **Multiple Format Support**: Supports various image formats (JPEG, PNG, etc.)

## Project Structure

```
face_final/
├── api.py                          # FastAPI web service
├── face_rec.py                     # Core face recognition module
├── gallery_manager.py              # Gallery management utilities
├── test.py                         # Test scripts
├── requirements.txt                # Python dependencies
├── checkpoints/                    # Model checkpoints
│   └── LightCNN_29Layers_V2_checkpoint.pth.tar
├── gallery/                        # Face galleries storage
│   ├── data/                       # Raw gallery data
│   └── galleries/                  # Processed gallery features
├── LightCNN/                       # LightCNN model implementation
│   ├── extract_features.py         # Feature extraction utilities
│   ├── light_cnn_v4.py            # LightCNN v4 model
│   ├── light_cnn.py               # Base LightCNN model
│   ├── load_imglist.py            # Image loading utilities
│   └── train.py                   # Training scripts
├── outputs/                        # Output images and results
└── yolo/                          # YOLO face detection
    ├── face_detec.py              # Face detection module
    └── weights/                   # YOLO model weights
        └── yolo11n-face.pt
```

## Installation

### Prerequisites

- Python 3.8+ (recommended: Python 3.10 or 3.11)
- CUDA-capable GPU (optional, for faster processing)
- Git

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd face_final
   ```
2. **Create and activate a Python environment**

   ```bash
   # Using conda (recommended)
   conda create -n face-rec python=3.10
   conda activate face-rec

   # Or using venv
   python -m venv face-rec
   source face-rec/bin/activate  # On Linux/Mac
   # face-rec\Scripts\activate  # On Windows
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```
4. **Download model checkpoints**

   ```bash
   mkdir -p checkpoints
   cd checkpoints

   # Download LightCNN checkpoint (replace with actual download link)
   wget "https://drive.google.com/open?id=1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS"
   # Or manually download and place the file in checkpoints/

   cd ..
   ```

### Troubleshooting Installation

If you encounter issues with numpy or other dependencies:

```bash
# For Python 3.12+ compatibility issues
pip install numpy>=1.25.0
pip install --upgrade setuptools wheel

# For CUDA-related issues
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Starting the API Server

```bash
python api.py
```

The API will be available at:

- **Main endpoint**: http://localhost:8000
- **Interactive documentation**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check

```http
GET /
```

Returns server status and basic information.

#### 2. Face Recognition

```http
POST /recognize
```

**Parameters:**

- `image`: Image file (multipart/form-data)
- `gallery_id`: Gallery ID to search against

**Response:**

```json
{
  "success": true,
  "matches": [
    {
      "person_id": "person_001",
      "confidence": 0.95,
      "similarity": 0.87
    }
  ],
  "processing_time": 0.234
}
```

#### 3. Gallery Management

```http
POST /gallery/create
PUT /gallery/{gallery_id}/add
GET /gallery/{gallery_id}/list
DELETE /gallery/{gallery_id}
```

### Python Module Usage

```python
from face_rec import FaceRecognizer
from gallery_manager import GalleryManager

# Initialize components
recognizer = FaceRecognizer()
gallery_manager = GalleryManager()

# Create a new gallery
gallery_manager.create_gallery("my_gallery")

# Add faces to gallery
gallery_manager.add_face("my_gallery", "person_001", "path/to/image.jpg")

# Recognize faces
results = recognizer.recognize("path/to/test_image.jpg", "my_gallery")
print(f"Recognition results: {results}")
```

## Configuration

### Model Configuration

Edit the model parameters in `face_rec.py`:

```python
# Face detection settings
DETECTION_CONFIDENCE = 0.5
DETECTION_NMS_THRESHOLD = 0.4

# Recognition settings
RECOGNITION_THRESHOLD = 0.6
MAX_FACES_PER_IMAGE = 10
```

### API Configuration

Modify `api.py` for server settings:

```python
# Server configuration
HOST = "0.0.0.0"
PORT = 8000
DEBUG = False

# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
```

## Testing

Run the test suite:

```bash
python test.py
```

### Manual Testing

Test individual components:

```bash
# Test face detection
python -c "from yolo.face_detec import detect_faces; print('Face detection working')"

# Test feature extraction
python -c "from LightCNN.extract_features import extract; print('Feature extraction working')"

# Test gallery operations
python -c "from gallery_manager import GalleryManager; gm = GalleryManager(); print('Gallery manager working')"
```

## Performance Optimization

### GPU Acceleration

Ensure CUDA is properly installed:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Memory Optimization

For large galleries or limited memory:

1. Enable batch processing in `face_rec.py`
2. Adjust `MAX_FACES_PER_IMAGE` parameter
3. Use image preprocessing to reduce file sizes

## Dependencies

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **OpenCV**: Computer vision library
- **NumPy**: Numerical computing
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation
- **Pillow**: Image processing library

## Model Information

### YOLO Face Detection

- **Model**: YOLOv11n-face
- **Input size**: 640x640
- **Performance**: ~100 FPS on modern GPUs

### LightCNN Feature Extraction

- **Architecture**: 29-layer LightCNN
- **Feature dimension**: 256
- **Training dataset**: Large-scale face datasets
