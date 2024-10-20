import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(pretrained=True, num_classes=2):
    model = models.resnet50(pretrained=pretrained)
    # Freeze early layers if needed
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
