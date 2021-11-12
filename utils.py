import os
import numpy as np
from collections import OrderedDict
from imageio import imsave, imread


def random_crop(image, mask=None, low_size=70):
    """Input image as a numpy array. Output a random crop from that image."""
    height = image.shape[0]
    width = image.shape[1]

    # get random image size
    rand_factor = np.random.randint(low=low_size, high=100) / 100
    rand_size = int(height * rand_factor)

    # select a random x and y pixel as starting point
    rand_y = np.random.randint(low=0, high=height - rand_size)
    rand_x = np.random.randint(low=0, high=width - rand_size)

    crop_image = image[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size, :]
    assert crop_image.shape[0] == crop_image.shape[1]

    if mask is not None:
        crop_mask = mask[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size]
        assert crop_image.shape[0] == crop_mask.shape[0]
        return crop_image, crop_mask
    else:
        return crop_image


def binarize_mask(mask):
    """Function to convert all masks to binary 0,1 labels"""

    # if three channel image
    if len(mask.shape) > 2:
        mask = mask.mean(axis=2)

    # binary value
    mask[mask > 0] = 1
    assert len(mask.shape) == 2, 'Incorrect mask size.'
    return mask


def preprocessing_rescale(img):
    """Simple rescaling function to make pixel values between 0-1"""
    if (np.max(img) > 1):
        return img / 255
    else:
        return img
    return img
