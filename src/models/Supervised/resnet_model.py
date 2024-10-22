import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def get_resnet_model(pretrained=True, num_classes=2):
    """
    Creates a ResNet50 model with backward compatible interface.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of output classes
    
    Returns:
        torch.nn.Module: Modified ResNet50 model
    """
    # Convert pretrained bool to weights parameter
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    
    # Initialize model with specified weights
    model = models.resnet50(weights=weights)
    
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model