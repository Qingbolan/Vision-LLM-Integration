import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

def get_vit_model(pretrained=True, num_classes=2):
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.vit_b_16(weights=weights)
    model.heads = nn.Linear(model.heads.head.in_features, num_classes)

    return model