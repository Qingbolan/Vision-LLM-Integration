import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scaling = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scaling
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

class UnsupervisedViTAnomalyDetector(nn.Module):
    def __init__(self, pretrained=True, latent_dim=512, 
                 img_size=224, patch_size=16, embed_dim=768,
                 num_heads=12, mlp_dim=3072, num_layers=6, dropout=0.1):
        super().__init__()
        
        # 特征提取器
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT) if pretrained else models.vit_b_16(weights=None)
        self.embed_dim = 768  # ViT-B/16 hidden dimension
        
        # 移除原始分类头
        self.vit.heads = nn.Identity()
        
        # 投影头 - 用于降维和特征提取
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, latent_dim),
            nn.Dropout(dropout)
        )
        
        # Transformer 编码器块
        self.encoder_blocks = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim=latent_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )
        
        # 用于计算重构误差的解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, self.embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 提取特征
        features = self.vit(x)  # [B, 768]
        
        # 投影到潜在空间
        z = self.projector(features)  # [B, latent_dim]
        
        # 通过编码器块
        z = self.encoder_blocks(z.unsqueeze(1)).squeeze(1)  # [B, latent_dim]
        
        # 重构特征
        reconstructed = self.decoder(z)  # [B, 768]
        
        return z, reconstructed
    
    def get_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            features = self.vit(x)
            z = self.projector(features)
            z = self.encoder_blocks(z.unsqueeze(1)).squeeze(1)
            reconstructed = self.decoder(z)
            
            # 计算重构误差
            reconstruction_error = F.mse_loss(reconstructed, features, reduction='none').mean(dim=1)
            
            # 计算潜在空间的范数
            z_norm = torch.norm(z, p=2, dim=1)
            
            # 组合异常评分
            anomaly_scores = reconstruction_error + 0.1 * z_norm
            
        return anomaly_scores

    def train_step(self, x, optimizer, device):
        self.train()
        x = x.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        features = self.vit(x)
        z = self.projector(features)
        z = self.encoder_blocks(z.unsqueeze(1)).squeeze(1)
        reconstructed = self.decoder(z)
        
        # 计算重构损失
        recon_loss = F.mse_loss(reconstructed, features)
        
        # 计算正则化损失
        reg_loss = 0.1 * torch.mean(torch.norm(z, p=2, dim=1))
        
        # 总损失
        total_loss = recon_loss + reg_loss
        
        # 反向传播
        total_loss.backward()
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return total_loss.item()
