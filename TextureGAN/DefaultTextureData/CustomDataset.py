import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_files = glob.glob(os.path.join(root_dir, '*.png'))
        self.transform = transform

        # Default transformation to convert images to tensors
        self.default_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')  # Convert image to RGB mode

        # Convert image to tensor
        image = self.default_transform(image)

        if self.transform:
            image = self.transform(image)

        return image


# Example of using the dataset with additional torchvision transforms
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

data_dir = './path_to_images_directory'
dataset = CustomDataset(root_dir=data_dir, transform=transform)
