import os

from TextureGAN.DefaultTextureData.CustomDataset import CustomDataset
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import Discriminator
import Generator

import time
import torch.nn as nn

# Create a random noise tensor for a batch of 8 samples
noise_batch = torch.randn(8, 100)

latent_dim=100
generator = Generator.Generator(latent_dim)
generated_images = generator.forward(noise_batch)
generated_images_numpy = generated_images.detach().cpu().numpy()
print(generated_images_numpy.shape)

discriminator = Discriminator.Discriminator()
v=discriminator.forward(generated_images)
print(v.detach().cpu().numpy())

print(generated_images_numpy)
Generator.plots(generated_images_numpy)

beta1 = 0.5
beta2 = 0.999
lr = 0.0002
# Optimizers
optimizer_G = optim.Adam(generator.parameters()\
                         , lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters()\
                         , lr=lr, betas=(beta1, beta2))
# Training loop
full_path = os.path.abspath(__file__)
base_path = os.path.dirname(full_path)
save_path="/Users/gametekker/Documents/ML/InterpretableGAN/used_data"
num_epochs=1000

"""
you are here
"""
print(save_path)
dataloader=DataLoader(CustomDataset(save_path))
# Loss function
adversarial_loss = nn.BCELoss()
for epoch in range(num_epochs):
    print(epoch)
    print(len(dataloader))
    for i, batch in enumerate(dataloader):
        # Convert list to tensor
        real_images = batch

        # Adversarial ground truths
        valid = torch.ones([1, 1, 1, 1])
        fake = torch.zeros([1, 1, 1, 1])

        # Configure input
        real_images = real_images.to()

        # ---------------------
        # Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(real_images.size(0), latent_dim)

        # Generate a batch of images
        fake_images = generator(z)

        # Measure discriminator's ability
        # to classify real and fake images
        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Backward pass and optimize
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_images = generator(z)

        # Adversarial loss
        g_loss = adversarial_loss(discriminator(gen_images), valid)

        # Backward pass and optimize
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Progress Monitoring
        # ---------------------

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]\
                        Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )

    # Save generated images for every epoch
    if (epoch + 1) % 50 == 0:
        Generator.makeGrid(generator(z).detach().cpu(),latent_dim,epoch)

