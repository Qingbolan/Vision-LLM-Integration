import torch
import torch.nn as nn
from torchvision import models

class DeepConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, encoded_space_dim=128):
        super(DeepConvolutionalVariationalAutoencoder, self).__init__()
        
        # 编码器部分，使用预训练的ResNet50作为基础
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去除最后两个层（平均池化和全连接层）
        
        # 添加自定义的编码层
        self.enc_fc = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1),  # 输入: [batch, 2048, 7, 7], 输出: [batch, 1024, 4, 4]
            nn.ReLU(True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),   # 输出: [batch, 512, 2, 2]
            nn.ReLU(True)
        )
        
        # 定义均值和对数方差的全连接层
        self.fc_mu = nn.Linear(512*2*2, encoded_space_dim)
        self.fc_logvar = nn.Linear(512*2*2, encoded_space_dim)
        
        # 解码部分
        self.decoder_fc = nn.Sequential(
            nn.Linear(encoded_space_dim, 512*2*2),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=0),  # 输出: [batch, 1024, 5, 5]
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),   # 输出: [batch, 512, 10, 10]
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),    # 输出: [batch, 256, 20, 20]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),    # 输出: [batch, 128, 40, 40]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),      # 输出: [batch, 3, 80, 80]
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.enc_fc(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 512, 2, 2)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar