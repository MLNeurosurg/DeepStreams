'''
1) Iteratively import images
2) Random crop from each image
3) Save each crop
'''
import os
import numpy as np
from collections import OrderedDict
from imageio import imsave, imread
# from skimage.io import imread
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser(description="Script that generates random crops from a directory of images and corresponding mask labels.")
parser.add_argument('-source_dir', type=str, help="Directory that contains labelled images and masks")
parser.add_argument("-save_dir", type=str, help="Directory to save images")
parser.add_argument('-crops_per_image', type=int, action='store_const', const=100, help="Number of crops to generate per image")
args = parser.parse_args()

IMAGE_SIZE = 256

def find_empty_mask(mask):
	# test all mask pixel black or white
	if (np.count_nonzero(mask) == 0) or (not(mask.flatten().all())):
		return True # meaning empty mask
	else:
		return False # keep this image

def random_crop(image, mask): # add mask to argument
	"""
	Input image as a numpy array
	Output a random crop from that image
	"""
	height = image.shape[0]
	width = image.shape[1]

	# while True:
	# select a random x and y pixel as starting point
	rand_factor = np.random.randint(low = 60, high = 100) / 100 # float between 0.4 and 1, want images no less than 40% of original
	rand_size = int(height * rand_factor)
	print(rand_size)

	rand_y = np.random.randint(low = 0, high = height - rand_size)
	rand_x = np.random.randint(low = 0, high = width - rand_size)

	crop_image = image[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size]
	crop_mask = mask[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size]

	assert crop_image.shape[0] == crop_image.shape[1]
	assert crop_image.shape[0] == crop_mask.shape[0]

	return crop_image, crop_mask

def random_crop_cell_mosaic(image, mask): # add mask to argument
	"""
	Input image as a numpy array
	Output a random crop from that image
	"""
	height = image.shape[0]
	width = image.shape[1]

	# rand_size
	rand_size = np.random.randint(low = 256, high = 1000)
	# rand_size = int(height * rand_factor)
	print(rand_size)

	rand_y = np.random.randint(low = 0, high = height - rand_size)
	rand_x = np.random.randint(low = 0, high = width - rand_size)

	crop_image = image[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size]
	crop_mask = mask[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size]

	assert crop_image.shape[0] == crop_image.shape[1]
	assert crop_image.shape[0] == crop_mask.shape[0]

	return crop_image, crop_mask

def import_image(image_dir, image_file):
	if 'png' in image_file or 'tif' in image_file:
		return imread(os.path.join(image_dir, image_file))
	else:
		print(image_file, "is not in directory.")

def preprocess_mask(mask):
	# binarize the mask using or logical statement
	# mask[np.logical_or((mask[:,:,0] > 0), (mask[:,:,1] > 0))] = 255
	bool_mask = mask[:,:,2] > 0
	bool_mask = np.repeat(bool_mask[:, :, np.newaxis], 3, axis=2)
	mask[bool_mask] = 255
	mask[~bool_mask] = 0
	return mask[:,:,0].reshape(mask.shape[0], mask.shape[1], 1) # index so that it

def preprocess_mask_cells(mask):
	# binarize the mask using or logical statement
	# mask[np.logical_or((mask[:,:,0] > 0), (mask[:,:,1] > 0))] = 255
	mask[mask > 0] = 255
	return mask

def crop_save_images(filename, number, image, mask):
	crop_image, crop_mask = random_crop(image, preprocess_mask(mask))

	crop_image = resize(crop_image, output_shape = (IMAGE_SIZE, IMAGE_SIZE), order = 3)
	crop_mask = resize(crop_mask, output_shape = (IMAGE_SIZE, IMAGE_SIZE), order = 3)

	imsave(args.save_dir + "/images/" + filename[0:-4] + "_" + str(number) + ".png", crop_image)
	imsave(args.save_dir + "/masks/" + filename[0:-4] + "_" + str(number) + ".png", crop_mask)

def crop_save_images_cells(filename, number, image, mask):
	crop_image, crop_mask = random_crop_cell_mosaic(image, preprocess_mask_cells(mask))

	crop_image = resize(crop_image, output_shape = (IMAGE_SIZE, IMAGE_SIZE), order = 3)
	crop_mask = resize(crop_mask, output_shape = (IMAGE_SIZE, IMAGE_SIZE), order = 3)

	imsave(args.save_dir + "/images/" + filename[0:-4] + "_" + str(number) + ".png", crop_image)
	imsave(args.save_dir + "/masks/" + filename[0:-4] + "_" + str(number) + ".png", crop_mask)


if __name__ == '__main__':

	# import filelist of masks and images, not that this relies on the masks and images have same alphanumeric sorted order
	image_list = sorted(os.listdir(os.path.join(args.source_dir, "images")))
	mask_list = sorted(os.listdir(os.path.join(args.source_dir, "masks")))

	# filter files by image
	image_list = [file for file in image_list if "png" in file or "tif" in file]
	mask_list = [file for file in mask_list if "png" in file or "tif" in file]
	images_masks = list(zip(image_list, mask_list))

	for image_mask_tuple in images_masks: # n_crop will be the nth crop from each image/mask
		print(image_mask_tuple)
		image_file, mask_file = image_mask_tuple # unpack tuple
		image = import_image(os.path.join(args.source_dir, "images"), image_file)
		mask = import_image(os.path.join(args.source_dir, "masks"), mask_file)

		for nth_crop in range(int(args.crops_per_image)):
			crop_save_images(image_file, nth_crop, image, mask)
