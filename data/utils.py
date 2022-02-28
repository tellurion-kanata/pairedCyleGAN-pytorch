import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image


def to_grayscale(data):
    grayscale = transforms.Grayscale(num_output_channels=3)(data)
    return grayscale


def draw_distribution(tensors, channel_num):
    count = 0
    for t in tensors:
        t = t.mean(dim=[2, 3]).squeeze(0).detach().cpu().numpy()
        x = np.linspace(0, channel_num, channel_num)
        plt.plot(x, t, label=str(count))
        count += 1
    plt.legend()
    plt.show()


def save_image(data, filename, grayscale=False):
    """
        image should be a torch.Tensor().cpu() [c, h, w]
        rgb value: [-1, 1] -> [0, 255]
    """

    img = (data.clone() + 1.) * 127.5

    img = img.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    if grayscale:
        img = img[:, :, 0]
    img = Image.fromarray(img)
    img.save(filename)
