import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import dlib

class LFWDataset(Dataset):
    def __init__(self, root_dir, pairs_file, transform=None, align_faces=False, predictor_path=None):
        self.root_dir = root_dir
        self.pairs_file = pairs_file
        self.transform = transform
        self.align_faces = align_faces
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector() if align_faces else None
        self.predictor = dlib.shape_predictor(predictor_path) if align_faces and predictor_path else None
        self.pairs = self._load_pairs(pairs_file)

    def _load_pairs(self, pairs_file):
        pairs = []
        # Read pairs.csv with header
        df = pd.read_csv(pairs_file, sep=',', header=0)
        
        print(f"Total rows in pairs.csv: {len(df)}")
        print(f"Columns in pairs.csv: {df.columns.tolist()}")
        for idx, row in df.iterrows():
            # Drop NaN values to count non-empty columns
            non_empty = row.dropna().tolist()
            print(f"Row {idx}: {non_empty}")
            if len(non_empty) == 3:  # Positive pair (name, imagenum1, imagenum2)
                person = str(row.iloc[0])  # name
                img1 = str(row.iloc[1])   # imagenum1
                img2 = str(row.iloc[2])   # imagenum2
                # Handle float image numbers (e.g., '1.0')
                try:
                    img1_num = int(float(img1.strip()))
                    img2_num = int(float(img2.strip()))
                except (ValueError, TypeError):
                    print(f"Skipping row {idx}: Invalid image numbers ({img1}, {img2})")
                    continue
                img1_path = os.path.join(self.root_dir, person, f"{person}_{img1_num:04d}.jpg")
                img2_path = os.path.join(self.root_dir, person, f"{person}_{img2_num:04d}.jpg")
                # Check if images exist (comment out for debugging)
                # if not os.path.exists(img1_path):
                #     print(f"Skipping row {idx}: Missing image {img1_path}")
                #     continue
                # if not os.path.exists(img2_path):
                #     print(f"Skipping row {idx}: Missing image {img2_path}")
                #     continue
                label = 1  # Same person
                print(f"Loaded positive pair: {person}, {img1}, {img2}, Paths: {img1_path}, {img2_path}")
            elif len(non_empty) == 4:  # Negative pair (name1, imagenum1, name2, imagenum2)
                person1 = str(row.iloc[0])  # name1
                img1 = str(row.iloc[1])    # imagenum1
                person2 = str(row.iloc[2]) # name2
                img2 = str(row.iloc[3])    # imagenum2
                # Handle float image numbers
                try:
                    img1_num = int(float(img1.strip()))
                    img2_num = int(float(img2.strip()))
                except (ValueError, TypeError):
                    print(f"Skipping row {idx}: Invalid image numbers ({img1}, {img2})")
                    continue
                img1_path = os.path.join(self.root_dir, person1, f"{person1}_{img1_num:04d}.jpg")
                img2_path = os.path.join(self.root_dir, person2, f"{person2}_{img2_num:04d}.jpg")
                # Check if images exist (comment out for debugging)
                # if not os.path.exists(img1_path):
                #     print(f"Skipping row {idx}: Missing image {img1_path}")
                #     continue
                # if not os.path.exists(img2_path):
                #     print(f"Skipping row {idx}: Missing image {img2_path}")
                #     continue
                label = 0  # Different people
                print(f"Loaded negative pair: {person1}, {img1}, {person2}, {img2}, Paths: {img1_path}, {img2_path}")
            else:
                print(f"Skipping row {idx}: Malformed row {non_empty}")
                continue
            pairs.append((img1_path, img2_path, label))
        print(f"Loaded {len(pairs)} valid pairs")
        return pairs

    def align_face(self, image):
        if not self.align_faces or not self.detector or not self.predictor:
            return image
        faces = self.detector(image, 1)
        if len(faces) == 0:
            return image  # Return original if no face detected
        shape = self.predictor(image, faces[0])
        left_eye = (shape.part(36).x, shape.part(36).y)  # Left eye (landmark 36)
        right_eye = (shape.part(45).x, shape.part(45).y)  # Right eye (landmark 45)
        
        # Calculate angle to align eyes horizontally
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (w, h))
        
        return aligned

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Image not found: {img1_path} or {img2_path}")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Apply face alignment if enabled
        if self.align_faces:
            img1 = self.align_face(img1)
            img2 = self.align_face(img2)
        
        if self.transform:
            img1 = self.transform(Image.fromarray(img1))
            img2 = self.transform(Image.fromarray(img2))
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((100, 100)),  # Resize to 100x100
        transforms.ToTensor(),  # Convert to tensor (normalizes to [0, 1])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])