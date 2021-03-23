import random
import torch
import cv2
import numpy as np


def image_from_gpu(img, to_send=False):
    if torch.cuda.is_available():
        img = img.cpu()
    if to_send:
        return cv2.cvtColor((img.detach().numpy().swapaxes(0, 2) * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    else:
        return img.detach().numpy().swapaxes(0, 2)


def random_hyperparameter_dict(input_dict):
    """
    You can use lists as values on dict, this
    function will pick one randomly at each iteration
    """
    chosen_params = {}

    for key in iter(input_dict):
        item = input_dict[key]
        chosen_params[key] = random.choice(
            item) if isinstance(item, list) else item
    return chosen_params


def stack_images_h(*images):
    for idx, image in enumerate(images):
        result = np.concatenate((result, image), axis=1) if idx else image
    return result


def stack_images_v(*images):
    for idx, image in enumerate(images):
        result = np.concatenate((result, image), axis=0) if idx else image
    return result
