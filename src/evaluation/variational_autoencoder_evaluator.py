import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
from src.training.variational_autoencoder_trainer import vae_loss_function


def evaluate_variational_autoencoder(model, dataloader, device, checkpoint_path='./checkpoints/', num_images=10):
    model.eval()
    total_loss = 0.0
    mse = nn.MSELoss()
    original_images = []
    reconstructed_images = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs, mu, logvar = model(inputs)
            loss = vae_loss_function(outputs, inputs, mu, logvar)
            total_loss += loss.item()

            if len(original_images) < num_images:
                original_images.append(inputs.cpu())
                reconstructed_images.append(outputs.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    print(f'Average Reconstruction Loss (VAE): {avg_loss:.4f}')

    # 可视化部分重构结果
    original = torch.cat(original_images)[:num_images]
    reconstructed = torch.cat(reconstructed_images)[:num_images]

    # 创建网格图像
    comparison = torch.cat([original, reconstructed])
    grid = make_grid(comparison, nrow=num_images, normalize=True)

    np_grid = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(20, 4))
    plt.imshow(np.clip(np_grid, 0, 1))
    plt.title('Original Images (Top) vs Reconstructed Images (Bottom)')
    plt.axis('off')
    plt.savefig(os.path.join(checkpoint_path, 'dcvae_reconstruction.png'))
    plt.close()
    print(f'Reconstruction images saved at {os.path.join(checkpoint_path, "dcvae_reconstruction.png")}')