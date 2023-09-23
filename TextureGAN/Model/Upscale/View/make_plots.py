import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np

def pngToTorch(dir):
    img = Image.open(dir).convert('RGB')
    t = transforms.ToTensor()
    img = t(img)
    img = img.unsqueeze(0)
    return (img)

def view_output(loc,generator,epoch):

    img = pngToTorch(loc)
    img = generator(img)
    grid = torchvision.utils.make_grid(img, nrow=4, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.savefig(f"/Users/gametekker/Documents/ML/InterpretableGAN/upscale/test/out{epoch}.pdf")