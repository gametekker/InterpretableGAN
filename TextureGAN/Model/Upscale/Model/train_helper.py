from TextureGAN.Model.Upscale.Data.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from SpecialGenerator import Generator
from SpecialDiscriminator import Discriminator
import torch.nn as nn
import torchvision.models as models
from train import train_loop

"""
you are here
"""
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features.eval()
        self.vgg19_truncated = nn.Sequential(*list(vgg19.children())[:9])  # Use up to 'relu2_2'
        for param in self.vgg19_truncated.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return nn.functional.mse_loss(self.vgg19_truncated(x), self.vgg19_truncated(y))

perceptual_loss_fn = VGGPerceptualLoss().to()

#process texture pack
#TODO: implement
features_dir = '/Users/gametekker/Documents/ML/InterpretableGAN/upscale/features'
labels_dir = '/Users/gametekker/Documents/ML/InterpretableGAN/upscale/labels'
dataloader=DataLoader(CustomDataset(features_dir,labels_dir), batch_size=10)

# Create generator, discriminator
generator=Generator(scaling_factor=2)
generator.initialize_with_srresnet("checkpoint_srresnet.pth.tar")

discriminator = Discriminator()
train_loop(generator,discriminator,dataloader,adversarial_loss=nn.BCEWithLogitsLoss(),percep_loss=perceptual_loss_fn)