import requests
import json

def test_face_recognition_api():
    # API endpoint
    url = "http://localhost:8000/recognize"
    
    # Prepare the data
    files = {
        'image': ('test_image.jpg', open('/home/spidey/Downloads/WhatsApp Image 2025-06-24 at 10.20.16 AM.jpeg', 'rb'), 'image/jpeg')
    }
    
    data = {
        'dept_id': ['101'],  # Multiple departments
        'year': ['2'],  # Corresponding years
        'section_students': ['714023247088', '714023247081', '714023247079', '714023247080', '714023247090']  # Student IDs
    }
    
    try:
        # Make the POST request
        response = requests.post(url, files=files, data=data)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Request successful!")
            print(f"📊 Students in section: {result['Students']}")
            print(f"👥 Detected students: {result['Detected Students']}")
            print(f"🔢 Total faces detected: {result['count']}")
            print(f"📷 Image returned: {'Yes' if result['image_base64'] else 'No'}")
            
            # Print detailed face detection results
            print("\n🎯 Face Detection Details:")
            for i, face in enumerate(result['faces']):
                print(f"  Face {i+1}: {face['identity']} (similarity: {face['similarity']:.3f})")
                print(f"    Bounding box: {face['bounding_box']}")
            
            # Optional: Save the annotated image
            if result['image_base64']:
                import base64
                img_data = base64.b64decode(result['image_base64'])
                with open('annotated_result.jpg', 'wb') as f:
                    f.write(img_data)
                print("\n💾 Annotated image saved as 'annotated_result.jpg'")
                
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the server is running on localhost:5564")
    except FileNotFoundError:
        print("❌ Test image file not found. Please update the file path.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        # Close the file
        if 'files' in locals():
            files['image'][1].close()

    """Test with a single department/year"""
    url = "http://localhost:5564/recognize"
    
    files = {
        'image': ('test_image.jpg', open('path/to/your/test_image.jpg', 'rb'), 'image/jpeg')
    }
    
    data = {
        'dept_id': ['CSE'],
        'year': ['2023'],
        'section_students': ['student123', 'student456', 'student789']
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single department test successful!")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Test failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        files['image'][1].close()

if __name__ == "__main__":
    print("🚀 Testing Face Recognition API...\n")
    
    # Update this path to your actual test image
    # test_image_path = "path/to/your/test_image.jpg"
    
    # Test 1: Multiple departments
    print("=== Test 1: Departments ===")
    test_face_recognition_api()
    
    print("\n" + "="*50 + "\n")
