import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

import warnings
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(context)
        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        z0 = self.ln1(x)
        attn_output = self.msa(z0)
        x = x + attn_output
        
        z1 = self.ln2(x)
        mlp_output = self.mlp(z1)
        x = x + mlp_output
        return x

class ViTAnomalyDetector(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, 
                 img_size=224, patch_size=16, embed_dim=768,
                 num_heads=12, mlp_dim=3072, num_layers=12):
        super().__init__()
        
        if pretrained:
            self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.embed_dim = 768  # ViT-B/16 hidden dimension
            
            # 重新创建分类头
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, num_classes)
            )
            
            # 替换原始的分类头
            self.vit.heads = self.classifier
        else:
            num_patches = (img_size // patch_size) ** 2
            self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
                for _ in range(num_layers)
            ])
            self.embed_dim = embed_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, num_classes)
            )
        
    def forward(self, x):
        if hasattr(self, 'vit'):
            # 直接使用ViT的前向传播
            # 这会自动处理patch embedding, position embedding, 
            # 和transformer layers
            x = self.vit(x)
            return x
        else:
            batch_size = x.shape[0]
            
            x = self.patch_embedding(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding
            
            for block in self.transformer_blocks:
                x = block(x)
                
            x = x[:, 0]
            x = self.classifier(x)
            return x

    def get_anomaly_score(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        anomaly_scores = 1 - probs[:, 0]  # 假设类别0为正常类
        return anomaly_scores