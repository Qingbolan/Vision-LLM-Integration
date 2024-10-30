# src/models/Unsupervised/autoencoder.py

import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, encoded_space_dim=128):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),    # Output: (64, H/2, W/2)
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Output: (128, H/4, W/4)
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # Output: (256, H/8, W/8)
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, encoded_space_dim), # Adjust according to image size
            nn.ReLU(True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 256 * 28 * 28),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 28, 28)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # Output: (128, H/4, W/4)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # Output: (64, H/2, W/2)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),    # Output: (3, H, W)
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
