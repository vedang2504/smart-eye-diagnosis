import os

print("🔥 Script started", flush=True)

base_path = r"D:\project\ML Project\smart-eye-diagnosis\data\raw"

print("Base path exists:", os.path.exists(base_path), flush=True)

splits = ['train', 'valid', 'test']

for split in splits:
    print(f"\n📂 {split.upper()} SET:", flush=True)

    split_path = os.path.join(base_path, split)
    print("Checking path:", split_path, flush=True)

    if not os.path.exists(split_path):
        print(f"❌ Path not found: {split_path}", flush=True)
        continue

    items = os.listdir(split_path)
    print("Items found:", items, flush=True)

    for cls in items:
        class_path = os.path.join(split_path, cls)

        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"👉 {cls}: {count}", flush=True)