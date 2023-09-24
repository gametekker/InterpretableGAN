import sys

print(sys.path)
from globals import config, hyperparameters
import zipfile
from PIL import Image
import PIL
import torch
from torchvision import transforms
from io import BytesIO
import os

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

# For a given mod (jar file), extract all relevant textures
def extract_png_tensors_from_jar(jar_path):
    images = []

    # Open the JAR as a ZIP file
    with zipfile.ZipFile(jar_path, 'r') as jar:
        # Iterate over the files in the JAR
        for name in jar.namelist():
            if name.endswith('.png'):
                with jar.open(name) as file:
                    # Convert the image to a PIL image
                    try:
                        img = Image.open(BytesIO(file.read()))
                        if img.size == (16, 16) and is_mostly_opaque(img):

                            if img.mode == 'RGBA':
                                img = img.convert('RGB')

                            # Convert the PIL image to a PyTorch tensor
                            to_tens = transforms.ToTensor()
                            img=to_tens(img)

                            images.append(img)

                    except PIL.UnidentifiedImageError:
                        continue

    # Stack all image tensors together
    return torch.stack(images)

def prepare_files(feature_pack_dir: str, label_pack_dir: str, resolution: int):
    data_dir=os.path.join(config()["project_dir"],"data_dir")
    os.makedirs(data_dir, exist_ok=True)

    os.makedirs(os.path.join(data_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels'), exist_ok=True)

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
    for (pack_dir, target_subdir, target_res) in [(feature_pack_dir, 'features', (16,16)), (label_pack_dir, 'labels', (resolution,resolution))]:
        # Walk through all files in the resource pack
        for dirpath, dirnames, filenames in os.walk(pack_dir):
            for file_name in filenames:
                full_path = os.path.join(dirpath, file_name)
                if full_path.endswith('.png'):
                    img_path = os.path.join(pack_dir, full_path)
                    img = Image.open(img_path)
                    print(img_path)
                    # Check if image resolution matches target - feature images must be mostly opaque
                    if img.size == target_res and (not target_res == (16,16) or is_mostly_opaque(img)):
                        # Save all 4 rotations
                        for angle in [0, 90, 180, 270]:
                            rotated_img = img.rotate(angle)
                            rotated_img_path = os.path.join(data_dir, target_subdir, f"{file_name}_{angle}.png")
                            rotated_img.save(rotated_img_path)
                            print(rotated_img_path)

    synchronize_directories(os.path.join(data_dir, 'features'),os.path.join(data_dir, 'labels'))