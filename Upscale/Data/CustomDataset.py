import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from globals import config, hyperparameters

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        data_dir=os.path.join(config()["project_dir"],"data_dir")
        features_dir=f"{data_dir}/features"
        labels_dir=f"{data_dir}/labels"

        self.features_files = glob.glob(os.path.join(features_dir, '*.png'))
        self.labels_files = glob.glob(os.path.join(labels_dir, '*.png'))
        assert check_same_resolution(self.features_files), "All feature textures must have same resolution"
        assert check_same_resolution(self.labels_files), "All label textures must have same resolution"
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

def check_same_resolution(features_files):
    if not features_files:  # Check if the list is empty
        return True

    # Get the resolution of the first image
    first_image = Image.open(features_files[0])
    first_resolution = first_image.size

    # Check if all other images have the same resolution
    for img_path in features_files[1:]:
        img = Image.open(img_path)
        if img.size != first_resolution:
            print(f"The image {img_path} has a different resolution of {img.size} compared to {first_resolution}.")
            return False

    return True