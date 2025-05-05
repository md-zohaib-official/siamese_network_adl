import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os

# Import Siamese Network
from model import SiameseNetwork

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define transformations (same as train.py and test.py)
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Adjust if train.py used different size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load('models/siamese_model.pth', map_location=device))
model.eval()

# Function to predict if two images are of the same person
def predict_same_person(img1_path, img2_path, model, transform, threshold=0.5):
    # Load and preprocess images
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
    except Exception as e:
        print(f"Error loading images: {e}")
        return None, None, None

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    # Get embeddings
    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(output1, output2).item()
    
    # Predict
    prediction = distance < threshold  # Closer than threshold = same person
    return img1, img2, (distance, prediction)

# Demo function to visualize and predict
def demo_face_verification(img1_path, img2_path):
    img1_tensor, img2_tensor, result = predict_same_person(img1_path, img2_path, model, transform)
    
    if result is None:
        print("Failed to process images.")
        return
    
    distance, prediction = result
    print(f"Distance: {distance:.4f}")
    print(f"Prediction: {'Same person' if prediction else 'Different people'}")

    # Visualize
    img1 = img1_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    img2 = img2_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(f'Image 2\nDistance: {distance:.4f}\n{"Same" if prediction else "Different"}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_prediction.png')
    plt.show()
    print("Prediction saved as 'demo_prediction.png'")

# Example usage
if __name__ == "__main__":
    # Example image paths (replace with your own)
    img1_path = 'dataset/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled/Abdullah_Gul/Abdullah_Gul_0001.jpg'
    img2_path = 'dataset/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled/Abdullah_Gul/Abdullah_Gul_0002.jpg'  # Same person
    # img2_path = 'dataset/lfw-deepfunneled/lfw-deepfunneled/AJ_Lamas/AJ_Lamas_0001.jpg'  # Different person

    demo_face_verification(img1_path, img2_path)