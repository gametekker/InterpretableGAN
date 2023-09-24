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

def plots(npa):

    # Display the images
    fig, axs = plt.subplots(1, 8, figsize=(15, 2))

    for i in range(8):
        img = npa[i].transpose(1, 2, 0)  # Convert from CxHxW to HxWxC for imshow
        axs[i].imshow(img,vmax=1,vmin=-1)
        axs[i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()


import torch.nn as nn

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # First upscale the image from 16x16 to 32x32
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Then use the SRCNN structure
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=9, padding=4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0), nn.ReLU())
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.upscale(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# Testing the generator
if __name__ == "__main__":
    generator = Generator()
    input_image = torch.randn((1, 3, 16, 16))  # Batch of 1, 3 channels, 16x16 image
    output_image = generator(input_image)
    print(output_image.shape)  # Expected: torch.Size([1, 3, 32, 32])


