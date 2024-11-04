import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

def get_vgg_model(pretrained=True, num_classes=2):
    if pretrained:
        weights = VGG16_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.vgg16(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    return model