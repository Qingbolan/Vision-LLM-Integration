import torch
import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, encoded_space_dim=128):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv layer: 3 -> 48 channels
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            
            # Second conv layer: 48 -> 96 channels
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            
            # Third conv layer: 96 -> 192 channels
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            
            # Fourth conv layer: 192 -> encoded_space_dim channels
            nn.Conv2d(192, encoded_space_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(encoded_space_dim),
            nn.ReLU(True)
        )
        
        # Decoder (symmetric to encoder)
        self.decoder = nn.Sequential(
            # First transposed conv: encoded_space_dim -> 192 channels
            nn.ConvTranspose2d(encoded_space_dim, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            
            # Second transposed conv: 192 -> 96 channels
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            
            # Third transposed conv: 96 -> 48 channels
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            
            # Fourth transposed conv: 48 -> 3 channels
            nn.ConvTranspose2d(48, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Final activation to ensure output is in [0,1]
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
    def get_reconstruction_error(self, x, threshold=None):
        """
        Calculate reconstruction error and optionally determine if it's an anomaly
        
        Args:
            x (torch.Tensor): Input image
            threshold (float, optional): Threshold for anomaly detection
            
        Returns:
            torch.Tensor: Reconstruction error
            bool: True if anomaly (if threshold provided)
        """
        reconstruction = self.forward(x)
        error = torch.mean((x - reconstruction) ** 2, dim=(1,2,3))
        
        if threshold is not None:
            return error, error > threshold
        return error