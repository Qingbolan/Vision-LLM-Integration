import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(256 * 28 * 28, encoded_space_dim),
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
            nn.ReLU(True),  # ReLU instead of Sigmoid at the last layer
        )
        
    def forward(self, x):
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)
        return x_reconstructed
