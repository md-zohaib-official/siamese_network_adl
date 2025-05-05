import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import LFWDataset, get_transforms
from model import SiameseNetwork, ContrastiveLoss

# Paths
root_dir = "dataset/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled"
pairs_file = "dataset/lfw-deepfunneled/pairs.csv"
predictor_path = "dataset/shape_predictor_68_face_landmarks.dat"

# Dataset
dataset = LFWDataset(
    root_dir,
    pairs_file,
    transform=get_transforms(),
    align_faces=False  # Set to True to test alignment
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = SiameseNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = ContrastiveLoss(margin=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for img1, img2, label in dataloader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), "models/siamese_model.pth")