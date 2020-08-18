# Mostly junk code left over that may end up being useful

import os
import numpy as np
from skimage.io import imread, imsave
import skimage.transform as trans

# def rotater(img, seed):
# 	np.random.seed(seed)
# 	rotate_angle = np.random.randint(365)
# 	return trans.rotate(img, rotate_angle, mode='reflect')

# def flipper(img, seed):
# 	np.random.seed(seed)
# 	flip = np.random.randint(0, 4)
# 	if flip == 0:
# 		return(np.fliplr(np.flipud(img)))
# 	if flip == 1:
# 		return(np.flipud(np.fliplr(img)))
# 	if flip == 2:
# 		return(np.flipud(img))
# 	if flip == 3:
# 		return(np.fliplr(img))

def flipper(img):
	img_1 = np.fliplr(np.flipud(img))
	img_2 = np.flipud(img)
	img_3 = np.fliplr(img)
	return img_1, img_2, img_3

def flipper(img):
	img_1 = np.fliplr(np.flipud(img))
	img_2 = np.flipud(img)
	img_3 = np.fliplr(img)
	return img_1, img_2, img_3

def data_augmenter(img_dir, mask_dir):
	# memoize the list of imported images

	files = sorted(os.listdir(img_dir)) # this move only works if the names of masks and images are the SAME
	
	for file in files:
		print(file)
		img = imread(img_dir + "/" + file)
		mask = imread(mask_dir + "/" + file)

		aug_imgs = flipper(img)
		aug_masks = flipper(mask)

		for i, aug_img in enumerate(aug_imgs):
			imsave(img_dir + "/" + file[0:-4] + "_aug_" + str(i) + ".png", aug_img.astype(np.uint8))
		
		for i, aug_mask in enumerate(aug_masks):
			imsave(mask_dir + "/" + file[0:-4] + "_aug_" + str(i) + ".png", aug_mask.astype(np.uint8))


if __name__ == '__main__':

	image_dir = "/home/todd/Desktop/oncostreams/training_tiles/images"
	mask_dir = "/home/todd/Desktop/oncostreams/training_tiles/masks"

	data_augmenter(image_dir, mask_dir)



