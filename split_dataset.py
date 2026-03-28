from PIL import Image
import os
import shutil
import random

def split_and_convert_folder(input_folder, output_base, train_pct=0.7, val_pct=0.15, target_size=(224, 224)):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * train_pct)
    n_val = int(n_total * val_pct)

    splits = {
        'train': files[:n_train],
        'val': files[n_train:n_train+n_val],
        'test': files[n_train+n_val:]
    }

    for split, split_files in splits.items():
        split_path = os.path.join(output_base, split, os.path.basename(input_folder))
        os.makedirs(split_path, exist_ok=True)
        for f in split_files:
            src = os.path.join(input_folder, f)
            # Always save as PNG if mode not compatible with JPEG
            ext = os.path.splitext(f)[1].lower()
            dst = os.path.join(split_path, f)
            try:
                img = Image.open(src)
                if img.mode == 'P':
                    img = img.convert('RGBA')
                img = img.resize(target_size)
                if ext in ['.jpg', '.jpeg'] and img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(dst)
            except Exception as e:
                print(f"Error processing {src}: {e}")

for class_name in ['with_mask', 'without_mask']:
    split_and_convert_folder(
        input_folder=os.path.join('dataset', class_name),
        output_base='split_dataset'
    )

print("Dataset split into train, val, and test folders. All images converted to 224x224.")
