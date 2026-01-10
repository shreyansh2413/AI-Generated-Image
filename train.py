print("TRAINING SCRIPT STARTED")

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import get_model

# Device (Mac = CPU)
device = "cpu"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = torchvision.datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# Load model
model = get_model().to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

# Save model
torch.save(model.state_dict(), "ai_image_detector.pth")
print("Training complete. Model saved.")
