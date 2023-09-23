import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features_files = glob.glob(os.path.join(features_dir, '*.png'))
        self.labels_files = glob.glob(os.path.join(labels_dir, '*.png'))
        assert len(self.features_files) == len(self.labels_files), "The two lists of images must have the same length"

        self.transform = transform

        # Default transformation to convert images to tensors
        self.default_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_name = self.features_files[idx]
        label_name = self.labels_files[idx]
        feature, label = Image.open(feature_name).convert('RGB'), Image.open(label_name).convert('RGB') # Convert image to RGB mode

        # Convert image to tensor
        feature, label = self.default_transform(feature), self.default_transform(label)

        if self.transform:
            feature, label = self.transform(feature), self.transform(label)

        return feature, label

from PIL import Image
import os

def prepare_files(data_dir: str, feature_pack_dir: str, label_pack_dir: str, resolution: int):
    # Ensure target directories exist
    os.makedirs(os.path.join(data_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels'), exist_ok=True)

    # Function to check if image has 100% opacity for all pixels
    def is_mostly_opaque(img: Image.Image) -> bool:
        if img.mode == "RGBA":
            total_pixels = img.size[0] * img.size[1]
            opaque_pixels = sum(1 for p in img.getdata() if p[3] == 255)
            return (opaque_pixels / total_pixels) > 0.75
        elif img.mode in ["RGB", "L"]:
            # If it's RGB or grayscale, it's implicitly fully opaque
            return True
        else:
            # For other image modes, you might need additional handling.
            # For now, return False or raise an exception for unsupported modes.
            return False

    def synchronize_directories(dir1: str, dir2: str):
        # Get filenames in each directory
        files_dir1 = set(os.listdir(dir1))
        files_dir2 = set(os.listdir(dir2))

        # Identify files that exist only in one directory
        only_in_dir1 = files_dir1 - files_dir2
        only_in_dir2 = files_dir2 - files_dir1

        # Remove the files that exist only in one directory
        for filename in only_in_dir1:
            os.remove(os.path.join(dir1, filename))

        for filename in only_in_dir2:
            os.remove(os.path.join(dir2, filename))

    # Process feature and label directories
    for (pack_dir, target_subdir) in [(feature_pack_dir, 'features'), (label_pack_dir, 'labels')]:
        for file_name in os.listdir(pack_dir):
            if file_name.endswith('.png'):
                img_path = os.path.join(pack_dir, file_name)
                img = Image.open(img_path)

                # Check if image resolution matches and is fully opaque
                if img.size == (resolution, resolution) or (img.size == (16, 16) and is_mostly_opaque(img)):
                    # Save all 4 rotations
                    for angle in [0, 90, 180, 270]:
                        rotated_img = img.rotate(angle)
                        rotated_img_path = os.path.join(data_dir, target_subdir, f"{file_name[:-4]}_{angle}.png")
                        rotated_img.save(rotated_img_path)
    synchronize_directories(os.path.join(data_dir, 'features'),os.path.join(data_dir, 'labels'))

features_dir = '/Users/gametekker/Downloads/VanillaDefault+1.20/assets/minecraft/textures/block'
labels_dir = '/Users/gametekker/Downloads/Faithful 32x - 1.20.1/assets/minecraft/textures/block'
data = '/Users/gametekker/Documents/ML/InterpretableGAN/upscale'
prepare_files(data, features_dir, labels_dir, 32)
