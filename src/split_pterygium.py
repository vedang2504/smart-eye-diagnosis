import os
import shutil
import random

source = r"D:\project\ML Project\smart-eye-diagnosis\data\raw\pterygium_all"
base_dest = r"D:\project\ML Project\smart-eye-diagnosis\data\raw"

images = os.listdir(source)
print("Total files found:", len(images))

random.shuffle(images)

train_split = int(0.7 * len(images))
val_split = int(0.85 * len(images))

for i, img in enumerate(images):
    src_path = os.path.join(source, img)

    if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        continue

    if i < train_split:
        folder = "train"
    elif i < val_split:
        folder = "valid"
    else:
        folder = "test"

    dest_path = os.path.join(base_dest, folder, "pterygium")
    os.makedirs(dest_path, exist_ok=True)

    shutil.copy(src_path, os.path.join(dest_path, img))

print("✅ Done splitting!")