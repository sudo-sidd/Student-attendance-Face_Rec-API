import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from LightCNN.light_cnn import LightCNN_29Layers_v2

# Consistent image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def load_model(model_path):
    """Load LightCNN model with correct architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for computation")
    
    # Initialize model with arbitrary number of classes (we only need embeddings)
    model = LightCNN_29Layers_v2(num_classes=100)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter out the fc2 layer parameters to avoid dimension mismatch
    if 'state_dict' in checkpoint:
        # Remove "module." prefix and fc2 layer parameters
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            # Skip fc2 layer parameters
            if 'fc2' in k:
                continue
            # Remove module prefix if present
            new_k = k.replace("module.", "")
            new_state_dict[new_k] = v
    else:
        # Direct state dict without 'state_dict' key
        new_state_dict = {}
        for k, v in checkpoint.items():
            if 'fc2' in k:
                continue
            new_k = k.replace("module.", "")
            new_state_dict[new_k] = v
    
    # Load the filtered state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    return model, device


def extract_embedding(model, img_path, device):
    """Extract a face embedding from an image using LightCNN"""
    try:
        # Load and transform image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            # LightCNN returns a tuple (output, features)
            _, embedding = model(img_tensor)
            return embedding.cpu().squeeze().numpy()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def create_gallery(model_path, data_dir, output_path):
    """Create a face recognition gallery from preprocessed face images"""
    # Load model
    model, device = load_model(model_path)
    
    # Create gallery dictionary
    gallery = {}
    
    # Process each identity folder
    identities = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(identities)} identities")
    
    for identity in identities:
        identity_dir = os.path.join(data_dir, identity)
        image_files = [f for f in os.listdir(identity_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"No images found for {identity}")
            continue
        
        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(identity_dir, img_file)
            embedding = extract_embedding(model, img_path, device)
            if embedding is not None:
                embeddings.append(embedding)
        
        if embeddings:
            # Average multiple embeddings for the same identity
            avg_embedding = np.mean(embeddings, axis=0)
            gallery[identity] = avg_embedding
            print(f"Added {identity} with {len(embeddings)} embeddings")
    
    # Save gallery
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"identities": gallery}, output_path)
    print(f"Gallery saved to {output_path} with {len(gallery)} identities")
    
    return gallery


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create face recognition gallery')
    parser.add_argument('--model_path', required=True, help='Path to LightCNN model')
    parser.add_argument('--data_dir', required=True, help='Directory with identity folders')
    parser.add_argument('--output_path', required=True, help='Output gallery file path')
    
    args = parser.parse_args()
    
    create_gallery(args.model_path, args.data_dir, args.output_path)
