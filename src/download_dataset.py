# Save as src/download_dataset.py
import kagglehub
import os
import shutil

# Download latest version of LFW dataset
path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

# Move dataset to dataset/lfw-deepfunneled
target_dir = "dataset/lfw-deepfunneled"
os.makedirs(target_dir, exist_ok=True)
for item in os.listdir(path):
    src_path = os.path.join(path, item)
    dst_path = os.path.join(target_dir, item)
    if os.path.isdir(src_path):
        shutil.move(src_path, dst_path)
    else:
        shutil.copy(src_path, dst_path)

print(f"Dataset downloaded and moved to: {target_dir}")