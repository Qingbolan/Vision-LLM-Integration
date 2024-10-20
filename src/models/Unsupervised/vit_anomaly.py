import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

class ViTAnomalyDetector(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(ViTAnomalyDetector, self).__init__()
        
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.vit = models.vit_b_16(weights=weights)
        # 修改分类头
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)