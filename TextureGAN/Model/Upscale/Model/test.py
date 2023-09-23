import torch
from TextureGAN.Model.Upscale.View import make_plots
net=torch.load("checkpoint_srresnet.pth.tar")["model"]
make_plots.view_output("/Users/gametekker/Documents/minecraftMods/ic2/profiles/classic/compactsolars/textures/blocks/high_voltage_bottom.png",net,4)