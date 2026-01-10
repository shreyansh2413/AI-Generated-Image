from PIL import Image
import os

FOLDER = "dataset/train/ai"

bad_files = []

for file in os.listdir(FOLDER):
    path = os.path.join(FOLDER, file)
    try:
        with Image.open(path) as img:
            img.verify()   # verify image integrity
    except Exception as e:
        bad_files.append(file)
        print("❌ Bad image:", file)

print("\nTotal bad images:", len(bad_files))

# OPTIONAL: delete bad images
for file in bad_files:
    os.remove(os.path.join(FOLDER, file))
    print("🗑 Deleted:", file)


