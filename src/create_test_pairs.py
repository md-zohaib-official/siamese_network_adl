import pandas as pd
import os
import sys

# Define paths (relative)
dataset_dir = './dataset/lfw-deepfunneled'
image_dir = os.path.join(dataset_dir, 'lfw-deepfunneled/lfw-deepfunneled')  # Images in dataset/lfw-deepfunneled/lfw-deepfunneled/
lfw_dir = './dataset/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled'  # Path prefix for test_pairs.csv
match_file = os.path.join(dataset_dir, 'matchpairsDevTest.csv')
mismatch_file = os.path.join(dataset_dir, 'mismatchpairsDevTest.csv')
output_file = './dataset/test_pairs.csv'

# List person directories
print("Person directories in dataset/lfw-deepfunneled/lfw-deepfunneled/:")
person_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
for d in sorted(person_dirs):
    print(f"  {d}")
print(f"Total person directories: {len(person_dirs)}")

# Check if CSV files exist
if not os.path.exists(match_file) or not os.path.exists(mismatch_file):
    print(f"Error: CSV files not found: {match_file}, {mismatch_file}")
    sys.exit(1)

# Load CSVs, force string type
try:
    match_df = pd.read_csv(match_file, dtype=str)  # name, imagenum1, imagenum2
    mismatch_df = pd.read_csv(mismatch_file, dtype=str)  # name, imagenum1, name.1, imagenum2
except Exception as e:
    print(f"Error reading CSV files: {e}")
    sys.exit(1)

# Debug: Print CSV info
print("matchpairsDevTest.csv columns:", match_df.columns.tolist())
print("matchpairsDevTest.csv sample row:", match_df.iloc[0].to_dict())
print("mismatchpairsDevTest.csv columns:", mismatch_df.columns.tolist())
print("mismatchpairsDevTest.csv sample row:", mismatch_df.iloc[0].to_dict())

# Function to get the nth image for a person
def get_image_path(name, image_num, image_dir):
    person_dir = os.path.join(image_dir, name)
    if not os.path.exists(person_dir):
        return None, f"Person directory not found: {person_dir}"
    
    # List .jpg files, sorted
    image_files = sorted([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
    if not image_files:
        return None, f"No images in {person_dir}"
    
    try:
        index = int(image_num) - 1  # 1-based to 0-based
        if index < 0 or index >= len(image_files):
            return None, f"Image index {image_num} out of range for {name} (max: {len(image_files)})"
        return os.path.join(name, image_files[index]), None
    except ValueError:
        return None, f"Invalid image number: {image_num}"

# Create match pairs (label=1)
match_pairs = []
skipped_matches = 0
for _, row in match_df.iterrows():
    name = row['name']
    img1_path, img1_error = get_image_path(name, row['imagenum1'], image_dir)
    img2_path, img2_error = get_image_path(name, row['imagenum2'], image_dir)
    
    if img1_error or img2_error:
        print(f"Skipping match pair for {name}: {img1_error or ''} {img2_error or ''}")
        skipped_matches += 1
        continue
    
    match_pairs.append({
        'img1': img1_path,
        'img2': img2_path,
        'label': 1
    })

# Create mismatch pairs (label=0)
mismatch_pairs = []
skipped_mismatches = 0
for _, row in mismatch_df.iterrows():
    name1, name2 = row['name'], row['name.1']
    img1_path, img1_error = get_image_path(name1, row['imagenum1'], image_dir)
    img2_path, img2_error = get_image_path(name2, row['imagenum2'], image_dir)
    
    if img1_error or img2_error:
        print(f"Skipping mismatch pair for {name1} vs {name2}: {img1_error or ''} {img2_error or ''}")
        skipped_mismatches += 1
        continue
    
    mismatch_pairs.append({
        'img1': img1_path,
        'img2': img2_path,
        'label': 0
    })

# Combine pairs
test_pairs = pd.DataFrame(match_pairs + mismatch_pairs)
if test_pairs.empty:
    print(f"Error: No valid pairs created. Skipped {skipped_matches} match pairs, {skipped_mismatches} mismatch pairs")
    sys.exit(1)

# Prepend lfw-deepfunneled/lfw-deepfunneled/
test_pairs['img1'] = test_pairs['img1'].apply(lambda x: os.path.join(lfw_dir, x))
test_pairs['img2'] = test_pairs['img2'].apply(lambda x: os.path.join(lfw_dir, x))

# Debug: Print sample pairs
print("Sample test_pairs rows:")
print(test_pairs.head().to_string())

# Save
try:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    test_pairs.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(test_pairs)} pairs")
except Exception as e:
    print(f"Error saving {output_file}: {e}")
    sys.exit(1)