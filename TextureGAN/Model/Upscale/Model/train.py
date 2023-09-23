import os
from TextureGAN.Model.Upscale.Data.CustomDataset import CustomDataset
import torch
import torchvision
import torch.optim as optim
from TextureGAN.Model.Upscale.View.make_plots import view_output

import numpy as np
import matplotlib.pyplot as plt

import time
import torch.nn as nn

def train_loop(generator, discriminator, dataloader, adversarial_loss=nn.BCELoss(),percep_loss=None):
    beta1 = 0.5
    beta2 = 0.999
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps")
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters()\
                             , lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters()\
                             , lr=lr, betas=(beta1, beta2))

    # Training loop
    full_path = os.path.abspath(__file__)
    base_path = os.path.dirname(full_path)
    save_path="/Users/gametekker/Documents/ML/InterpretableGAN/used_data"
    num_epochs=10000

    # Prepare
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    adversarial_loss = adversarial_loss.to(device)

    for epoch in range(num_epochs):
        upscaled=None
        for i, (feature, label) in enumerate(dataloader):

            feature=feature.to(device)
            label=label.to(device)

            # Adversarial ground truths
            valid = torch.ones([feature.shape[0], 1, 1, 1])
            fake = torch.zeros([feature.shape[0], 1, 1, 1])

            upscaled = generator(feature)

            # ---------------------
            # Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(label), valid)
            fake_loss = adversarial_loss(discriminator(upscaled.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # Backward pass and optimize
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Calculate adversarial loss
            g_loss = adversarial_loss(discriminator(upscaled), valid)

            # Augment with perceptual loss if defined
            if percep_loss is not None:
                g_loss = 0.1*g_loss + percep_loss(upscaled, label)

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
        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                view_output("/Users/gametekker/Documents/ML/InterpretableGAN/upscale/test/polished_basalt_top.png",generator,epoch)
                torch.save({'epoch': epoch,
                            'model': generator,
                            'discrim': discriminator},
                           'checkpoint_trainloop.pth.tar')