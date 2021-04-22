import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

def adjustData(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.] = 1
    mask = mask.clip(min = 0, max = 1)
    return img, mask

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "rgb",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                    num_class = 2, save_to_dir = None, target_size = (256,256), seed = 1):
    '''
    Will generate image and mask at the same time.
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        interpolation = "bicubic",
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        interpolation = "bicubic",
        seed = seed)

    # combine the the two ImageDataGenerators into single generator to yield image, masks pairs
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img, mask)
        yield (img,mask)