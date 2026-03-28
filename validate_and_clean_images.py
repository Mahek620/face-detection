from PIL import Image
import os

def validate_images(folder):
    count = 0
    for root, dirs, files in os.walk(folder):
        for name in files:
            img_path = os.path.join(root, name)
            try:
                img = Image.open(img_path)
                img.verify()  # PIL will raise an exception for corrupt files
            except Exception as e:
                print(f"Invalid image: {img_path} ({e})")
                os.remove(img_path)
                count += 1
    print(f"Removed {count} invalid or corrupt images.")

validate_images('split_dataset/train')
validate_images('split_dataset/val')
validate_images('split_dataset/test')
