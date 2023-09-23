import os

from TextureGAN.DefaultTextureData.CustomDataset import CustomDataset
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import Discriminator

import time
import torch.nn as nn

def makeGrid(generated,latent_dim,epoch):
    with torch.no_grad():
        z = torch.randn(16, latent_dim)
        grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis("off")
        plt.savefig(f"/Users/gametekker/Documents/ML/InterpretableGAN/used_data/outs/out{epoch}.pdf")

def plots(npa):

    # Display the images
    fig, axs = plt.subplots(1, 8, figsize=(15, 2))

    for i in range(8):
        img = npa[i].transpose(1, 2, 0)  # Convert from CxHxW to HxWxC for imshow
        axs[i].imshow(img,vmax=1,vmin=-1)
        axs[i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),  # [n, 64, 4, 4]
            nn.Linear(256 * 4 * 4, 128 * 4 * 4),
            nn.Linear(128 * 4 * 4, 64 * 4 * 4),
            nn.Unflatten(1,(64,4,4)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # [n, 64, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2),  # [n, 16, 34, 34]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=5)  # [n, 1, 28, 28]
        )

    def forward(self, x):
        return self.model(x)

