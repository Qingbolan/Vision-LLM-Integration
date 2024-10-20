import torch.nn as nn
from torchvision import models

class DeepConvolutionalAutoencoder(nn.Module):
    def __init__(self, encoded_space_dim=128):
        super(DeepConvolutionalAutoencoder, self).__init__()
        
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
        
        # 编码后的全连接层
        self.fc1 = nn.Linear(512*2*2, encoded_space_dim)

        # 解码部分
        self.decoder_fc = nn.Sequential(
            nn.Linear(encoded_space_dim, 512*2*2),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=0),  # 输出: [batch, 1024, 5, 5]
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: [batch, 512, 10, 10]
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),   # 输出: [batch, 256, 20, 20]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   # 输出: [batch, 128, 40, 40]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),     # 输出: [batch, 3, 80, 80]
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.enc_fc(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 512, 2, 2)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded