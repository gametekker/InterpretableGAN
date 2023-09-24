import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features.eval()
        self.vgg19_truncated = nn.Sequential(*list(vgg19.children())[:9])  # Use up to 'relu2_2'
        for param in self.vgg19_truncated.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return nn.functional.mse_loss(self.vgg19_truncated(x), self.vgg19_truncated(y))