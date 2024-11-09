import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights
import torch.nn.functional as F
import math
import os

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
    def __init__(self, pretrained=True, latent_dim=768, 
                 img_size=224, patch_size=16, embed_dim=768,
                 num_heads=12, mlp_dim=3072, num_layers=6, dropout=0.1):
        super().__init__()
        
        # 特征提取器
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT) if pretrained else models.vit_b_16(weights=None)
        self.embed_dim = embed_dim  # 使用传入的 embed_dim 参数
        
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
            nn.Linear(mlp_dim, 512 * 7 * 7),  # 输出到卷积输入大小
            nn.GELU(),
            # 使用卷积层上采样
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 将输出限制在 [-1, 1]
        )
        
        self.output_shape = (3, img_size, img_size)

    def forward(self, x):
        # 提取特征
        features = self.vit(x)  # [B, embed_dim]
        
        # 投影到潜在空间
        z = self.projector(features)  # [B, latent_dim]
        
        # 通过编码器块
        z = self.encoder_blocks(z.unsqueeze(1)).squeeze(1)  # [B, latent_dim]
        
        # 重构特征
        reconstructed = self.decoder(z.view(-1, 512, 7, 7))  # 调整到卷积输入形状
        reconstructed = reconstructed.view(-1, *self.output_shape)  # [B, 3, img_size, img_size]
        
        return z, reconstructed

# 加载训练好的模型权重
pretrained_path = "checkpoints/best_vit_anomaly.pth"
checkpoint = torch.load(pretrained_path, weights_only=True)

# 初始化新模型，确保 latent_dim 和 embed_dim 与训练时一致
model = UnsupervisedViTAnomalyDetector(pretrained=True, latent_dim=768, embed_dim=768, num_heads=12)
model_dict = model.state_dict()

# 过滤掉解码器中新增层的权重
pretrained_weights = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

# 更新新模型的 state_dict
model_dict.update(pretrained_weights)
model.load_state_dict(model_dict)

# 冻结已经训练好的权重（如果不想重新训练整个模型）
for name, param in model.named_parameters():
    if 'decoder' not in name:
        param.requires_grad = False

# 保存 finetune 后的模型，带有 'model_state_dict' 键
finetuned_model_path = "checkpoints/finetuned_vit_anomaly.pth"
torch.save({'model_state_dict': model.state_dict()}, finetuned_model_path)
print(f"Finetuned model saved to: {finetuned_model_path}")
