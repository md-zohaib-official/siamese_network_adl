import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import Siamese Network
from model import SiameseNetwork  # model.py is in src/

# Set device (MPS for MacBook)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Adjust based on train.py
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class LFWPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_file, root_dir, transform=None):
        self.pairs = pd.read_csv(pairs_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_dir, self.pairs.iloc[idx]['img1'])
        img2_path = os.path.join(self.root_dir, self.pairs.iloc[idx]['img2'])
        label = self.pairs.iloc[idx]['label']

        if not os.path.exists(img1_path):
            raise FileNotFoundError(f"Image not found: {img1_path}")
        if not os.path.exists(img2_path):
            raise FileNotFoundError(f"Image not found: {img2_path}")

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# Load test dataset
test_dataset = LFWPairsDataset(
    pairs_file='./dataset/test_pairs.csv',
    root_dir='./',  # Absolute path
    transform=transform
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load('./models/siamese_model.pth'))
model.eval()

# Evaluation
distances = []
labels = []
predictions = []
threshold = 0.5  # Adjust after initial run

with torch.no_grad():
    for img1, img2, label in tqdm(test_loader, desc="Testing"):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        # Get embeddings
        output1, output2 = model(img1, img2)
        
        # Compute Euclidean distance
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        distances.extend(distance.cpu().numpy())
        labels.extend(label.cpu().numpy())
        
        # Predict based on threshold
        pred = (distance < threshold).float()
        predictions.extend(pred.cpu().numpy())

# Convert to numpy arrays
distances = np.array(distances)
labels = np.array(labels)
predictions = np.array(predictions)

# Compute metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(labels, -distances)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./roc_curve.png')
plt.close()

# Visualize sample predictions
def visualize_predictions(dataset, model, num_samples=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    plt.figure(figsize=(15, 5))
    
    for i, idx in enumerate(indices):
        img1, img2, label = dataset[idx]
        img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output1, output2 = model(img1, img2)
            distance = torch.nn.functional.pairwise_distance(output1, output2).item()
            pred = distance < threshold
        
        img1 = img1.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        img2 = img2.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img1)
        plt.title(f'Image 1\nLabel: {"Same" if label else "Different"}')
        plt.axis('off')
        
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(img2)
        plt.title(f'Image 2\nDist: {distance:.2f}\nPred: {"Same" if pred else "Different"}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./sample_predictions.png')
    plt.close()

visualize_predictions(test_dataset, model)
print("Sample predictions saved as './sample_predictions.png'")