import requests
import json
import os

def test_face_recognition_api():
    """Simple test for the face recognition API"""
    
    # API endpoint
    url = "http://localhost:8000/recognize"
    
    # Test image path - replace with your test image
    test_image_path = "test_image.jpg"  # Put your test image here
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please place a test image named 'test_image.jpg' in the current directory")
        return
    
    # Prepare the request
    files = {
        'image': ('test_image.jpg', open(test_image_path, 'rb'), 'image/jpeg')
    }
    
    data = {
        'gallery_name': 'default'  # Use your gallery name
    }
    
    try:
        # Make the POST request
        response = requests.post(url, files=files, data=data)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Request successful!")
            print(f"📊 Success: {result['success']}")
            print(f"👥 Faces detected: {result['faces_detected']}")
            
            # Print detailed face detection results
            print("\n🎯 Face Recognition Results:")
            for i, face in enumerate(result['faces']):
                print(f"  Face {i+1}: {face['identity']} (confidence: {face['confidence']:.3f})")
                print(f"    Bounding box: {face['bounding_box']}")
                
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'files' in locals():
            files['image'][1].close()


def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")


if __name__ == "__main__":
    print("🧪 Testing Face Recognition API\n")
    
    # Test health check first
    print("1. Testing health check...")
    test_health_check()
    
    print("\n2. Testing face recognition...")
    test_face_recognition_api()
