import os
import shutil
import random

print("🔥 Splitting Pterygium Dataset...")

# 🔹 Source (your current folder)
source = r"D:\project\ML Project\smart-eye-diagnosis\data\raw\pterygium"

# 🔹 Destination
base_dest = r"D:\project\ML Project\smart-eye-diagnosis\data\raw"

# 🔹 Get all images
images = os.listdir(source)

# Filter only image files
images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

print("Total images found:", len(images))

# Shuffle
random.shuffle(images)

# Split ratios
train_split = int(0.7 * len(images))
val_split = int(0.85 * len(images))

for i, img in enumerate(images):
    src_path = os.path.join(source, img)

    if i < train_split:
        folder = "train"
    elif i < val_split:
        folder = "valid"
    else:
        folder = "test"

    dest_path = os.path.join(base_dest, folder, "pterygium")
    os.makedirs(dest_path, exist_ok=True)

    shutil.copy(src_path, os.path.join(dest_path, img))

print("✅ Pterygium split completed!")