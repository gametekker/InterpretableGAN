# Torch
import torch

# Dataset/DataLoader
from Data.CustomDataset import CustomDataset
from torch.utils.data import DataLoader

# Generator, Discriminator
from ResNetImplementation.SpecialGenerator import Generator
from ResNetImplementation.SpecialDiscriminator import Discriminator

from ResNetImplementation.train import train_loop, train_generator
from ResNetImplementation.VGGPerceptualLoss import VGGPerceptualLoss

def PerformTraining(config,hyperparameters,experimentlogger):

    """
    Load configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else config["device"])
    scaling_factor = config["scaling_factor"]
    n_blocks = config["n_blocks"]

    """
    Load hyperparameters
    """

    generator_only_epochs=hyperparameters["generator_only_epochs"]
    batch_size=hyperparameters["batch_size"]
    epochs=hyperparameters["epochs"]
    beta1 = hyperparameters["beta1"]
    beta2 = hyperparameters["beta2"]
    learning_rate = hyperparameters["learning_rate"]
    adversarial_loss_weight = hyperparameters["adversarial_loss_weight"]

    """
    ResNetImplementation components
    """

    # Set device
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    # Initialize generator
    generator=Generator(scaling_factor=scaling_factor)

    # Initialize discriminator - note image size before pooling = resolution / (2^(n_blocks/2))
    discriminator = Discriminator(n_blocks=n_blocks)

    # Initialize dataloader
    dataloader=DataLoader(CustomDataset(), batch_size=batch_size)

    # Initialize loss function for adversarial loss
    adversarial_loss = torch.nn.BCEWithLogitsLoss()

    # Define loss function that augments perceptual loss with annealed adversarial loss value
    # required inputs:
    # - adversarial_loss: adversarial loss
    # - upscaled: upscaled image from generator
    # - label: ground truth image
    # - epoch: current training epoch
    # - num_epochs: total number of training epochs
    augmented_loss = lambda adversarial_loss, upscaled, label, epoch, num_epochs: (adversarial_loss * (epoch/num_epochs)) * adversarial_loss_weight + perceptual_loss_fn(upscaled, label)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    """
    Train
    """

    # Train the generator without adversarial
    train_generator(experimentlogger,generator,dataloader,optimizer_G,loss=perceptual_loss_fn,device=device, end_epoch=generator_only_epochs)

    # Train the generator, discriminator
    train_loop(experimentlogger,generator,discriminator,dataloader,optimizer_G,optimizer_D,adversarial_loss=adversarial_loss,total_loss=augmented_loss,device=device, end_epoch=epochs)
