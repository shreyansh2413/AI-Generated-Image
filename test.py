import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
from model import get_model

# Device
device = torch.device("cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("ai_image_detector.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Labels
labels = {0: "AI-GENERATED", 1: "REAL"}

# Folder path
folder_path = "test_images"

if not os.path.exists(folder_path):
    print("❌ test_images folder not found")
    sys.exit()

print("\n🔍 Batch Testing Started...\n")

for file in os.listdir(folder_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(folder_path, file)

        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image)
                probs = F.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item() * 100

            print(f"{file:25} → {labels[pred]}  ({confidence:.2f}%)")

        except Exception as e:
            print(f"{file} → ❌ Error reading image")

print("\n✅ Batch testing complete\n")

