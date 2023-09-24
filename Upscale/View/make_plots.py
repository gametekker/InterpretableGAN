from PIL import Image
import torch
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os

def pngToTorch(dir):
    img = Image.open(dir).convert('RGB')
    t = transforms.ToTensor()
    img = t(img)
    img = img.unsqueeze(0)
    return (img)

def view_output(img,generator,epoch):
    img = generator(img)
    grid = torchvision.utils.make_grid(img, nrow=4, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.savefig(f"/Users/gametekker/Documents/ML/InterpretableGAN/upscale/test/out{epoch}.pdf")


def process_and_save_images_as_pdf(tensor_images, model, save_path):
    """
    For each image in the tensor, process it through the model and save the original and the output as a PDF.

    :param tensor_images: A tensor containing the images in CxHxW format
    :param model: The pre-trained model to process the images
    :param save_path: The path to save the PDF files
    :return: None
    """

    # Ensure the model is in eval mode
    model.eval()

    # Define a transformation to convert PyTorch tensor back to PIL Image
    to_pil = transforms.ToPILImage()

    for idx, image_tensor in enumerate(tensor_images):
        with torch.no_grad():
            output_tensor = model(image_tensor.unsqueeze(0))  # Add batch dimension and pass through the model

        # Convert tensors to PIL Images for visualization
        original_img = to_pil(image_tensor)
        output_img = to_pil(output_tensor.squeeze(0))

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(output_img)
        ax2.set_title('Processed Image')
        ax2.axis('off')

        pdf_filename = f"{save_path}/output_{idx}.pdf"
        plt.savefig(pdf_filename)
        plt.close()

    print("Processing complete. PDF files saved.")
