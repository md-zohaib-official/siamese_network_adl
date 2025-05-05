from utils import LFWDataset, get_transforms
from torch.utils.data import DataLoader

# Paths
root_dir = "dataset/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled"  # Corrected to point to person folders
pairs_file = "dataset/lfw-deepfunneled/pairs.csv"
predictor_path = "dataset/shape_predictor_68_face_landmarks.dat"

# Initialize dataset
dataset = LFWDataset(
    root_dir,
    pairs_file,
    transform=get_transforms(),
    align_faces=True,  # Enable alignment
    predictor_path=predictor_path
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test one batch
try:
    for img1, img2, label in dataloader:
        print(f"Image 1 shape: {img1.shape}")  # Should be [4, 3, 100, 100]
        print(f"Image 2 shape: {img2.shape}")
        print(f"Labels: {label}")
        break
except Exception as e:
    print(f"Error: {e}")