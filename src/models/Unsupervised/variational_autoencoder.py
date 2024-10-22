import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalVAE(nn.Module):
    def __init__(self, encoded_space_dim=128):
        super(ConvolutionalVAE, self).__init__()
        
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
            nn.ReLU(True)
        )
        
        # FC layers for mean and variance
        self.fc_mu = nn.Linear(192 * 28 * 28, encoded_space_dim)
        self.fc_var = nn.Linear(192 * 28 * 28, encoded_space_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(encoded_space_dim, 192 * 28 * 28)
        
        # Decoder (symmetric to encoder)
        self.decoder = nn.Sequential(
            # First transposed conv: 192 -> 96 channels
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            
            # Second transposed conv: 96 -> 48 channels
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            
            # Third transposed conv: 48 -> 3 channels
            nn.ConvTranspose2d(48, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        # Encode input
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get mean and variance
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        # Decode latent vector
        x = self.decoder_input(z)
        x = x.view(x.size(0), 192, 28, 28)  # Reshape
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def get_reconstruction_error(self, x, threshold=None):
        """
        Calculate reconstruction error and KL divergence
        """
        recon_x, mu, log_var = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='none').sum(dim=(1,2,3))
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # Total loss
        total_loss = recon_loss + kl_div
        
        if threshold is not None:
            return total_loss, total_loss > threshold
        return total_loss

def vae_loss_function(recon_x, x, mu, log_var):
    """
    VAE loss function = Reconstruction loss + KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_div