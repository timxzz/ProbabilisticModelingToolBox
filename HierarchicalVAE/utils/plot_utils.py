import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.utils

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def show_reconstruct_image(image, model, device):
    with torch.no_grad():
        image = image.to(device)
        image = model.reconstruction(image)
        image = image.cpu()
        image = to_img(image)
        show_image(image[0])

def visualise_output(images, model, device):

    with torch.no_grad():
    
        images = images.to(device)
        images = model.reconstruction(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()