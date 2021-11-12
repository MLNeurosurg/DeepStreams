import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import Stream_Dataset
from unet import UNet

data_root = Path('/home/labcomputer/Desktop/oncostreams/data')
images_path = data_root / 'images'
masks_path = data_root / 'masks'

def view_image_mask(image, mask):
    image = image.numpy()
    mask = mask.numpy()
    img = np.moveaxis(image, 0, -1)
    mask = np.moveaxis(mask, 0, -1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title("Oncostream image")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(mask, cmap='viridis')
    ax2.set_title("mask")
    plt.show()


# get dataloader
dataset = Stream_Dataset(images_path, masks_path, image_size=256)
train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

# import model
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
net = UNet(n_channels=3, n_classes=1)
net.to(device)

# optimizer and learn rate schedule
optimizer = optim.Adam(net.parameters(), lr=0.000005, weight_decay=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
criterion = nn.BCEWithLogitsLoss()

n_epochs = 20
for epoch in range(n_epochs):
	print("=============={}===============".format(epoch+1))
	net.train()
	batch_idx = 1
	epoch_loss = 0
	correct = 0
	total = 0
	for images, masks in train_loader:
		images = images.to(device=device).float()
		masks  = masks.to(device=device).float()
		
		masks_pred = net(images)
		assert masks_pred.shape == masks.shape
		loss = criterion(masks_pred, masks)
		epoch_loss += loss.item()

		# pred = pred.cpu()
		preds = torch.ge(torch.sigmoid(masks_pred), 0.5).float()
		correct += (preds == masks).sum().cpu().numpy()
		total += (masks.size(0) * masks.size(1) * masks.size(2) * masks.size(3))
		print("Iteration: " + str(batch_idx) + " >>>> epoch accuracy: " + str(correct/total), 
					end = '\r', flush = True)

		batch_idx += 1

		optimizer.zero_grad()
		loss.backward()
		# nn.utils.clip_grad_value_(net.parameters(), 0.1)
		optimizer.step()
