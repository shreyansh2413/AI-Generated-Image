import torch.nn as nn
from torchvision import models

def get_model():
    # Load pre-trained ResNet18
    model = models.resnet18(weights=None)

    # Replace final layer for 2 classes (AI vs REAL)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model
