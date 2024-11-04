# src/models/Unsupervised/variational_autoencoder.py

import torch
import torch.nn as nn

class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, encoded_space_dim=128):
        super(ConvolutionalVariationalAutoencoder, self).__init__()
        
        self.encoded_space_dim = encoded_space_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),    # Output: (64, H/2, W/2)
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Output: (128, H/4, W/4)
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # Output: (256, H/8, W/8)
            nn.ReLU(True),
        )
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 28 * 28, encoded_space_dim)
        self.fc_logvar = nn.Linear(256 * 28 * 28, encoded_space_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(encoded_space_dim, 256 * 28 * 28)
        
        self.decoder_conv = nn.Sequential(
            nn.ReLU(True),
            nn.Unflatten(1, (256, 28, 28)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # Output: (128, H/4, W/4)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # Output: (64, H/2, W/2)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),    # Output: (3, H, W)
            nn.ReLU(True),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder_fc(z)
        x = self.decoder_conv(x)
        return x, mu, logvar
