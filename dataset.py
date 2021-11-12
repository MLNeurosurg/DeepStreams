import os
import random
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import random_crop, binarize_mask
from typing import List, Dict
from imageio import imread


class Stream_Dataset(Dataset):
	def __init__(self, image_root: str,
					mask_root: str,
					image_size: int = 256,
					lowest_perc_size: int = 70,
					testing: bool = False) -> Dict:
		
		self.image_root = Path(image_root)
		self.mask_root = Path(mask_root)
		self.image_size = image_size
		self.lowest_perc_size = lowest_perc_size
		self.testing = testing

		# repeat the lists
		self.imagefilelist = sorted(os.listdir(image_root)) * 10
		self.maskfilelist = sorted(os.listdir(mask_root)) * 10


	def transform(self, image, mask): 

		pil = transforms.ToPILImage()
		image = TF.resize(pil(image), size = self.image_size)
		mask = TF.resize(pil(mask), size = self.image_size)
		
		# Random horizontal flipping
		if random.random() > 0.5:
			image = TF.hflip(image)
			mask = TF.hflip(mask)

		# Random vertical flipping
		if random.random() > 0.5:
			image = TF.vflip(image)
			mask = TF.vflip(mask)

		if random.random() > 0.3:
			rand_val = random.choice(np.arange(0.25, 1.75, 0.1).tolist())
			image = TF.adjust_brightness(image, brightness_factor = rand_val)

		if random.random() > 0.3:
			rand_val = random.choice(np.arange(0.25, 1.75, 0.1).tolist())
			image = TF.adjust_saturation(image, saturation_factor = rand_val)

		if random.random() > 0.3:
			rand_val = random.choice(np.arange(0.25, 1.75, 0.1).tolist())
			image = TF.adjust_contrast(image, contrast_factor = rand_val)

		image = TF.to_tensor(image)
		mask = TF.to_tensor(mask)
		mask[mask > 0] = 1

		return image, mask

	def __len__(self):
		return len(self.imagefilelist)

	def __getitem__(self, idx):
		
		# get image id and load image
		image_file = self.imagefilelist[idx]
		mask_file = self.maskfilelist[idx]
		image = imread(self.image_root / image_file)[:,:,0:3]
		mask = imread(self.mask_root / mask_file)

		# get random crop from image
		image, mask = random_crop(image, mask, low_size=self.lowest_perc_size)
		mask = binarize_mask(mask)

		image, mask = self.transform(image.astype(np.uint8), mask.astype(np.uint8))
		return image, mask, image_file, mask_file   

