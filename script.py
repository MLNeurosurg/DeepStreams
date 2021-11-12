
'''
1) feedforward
2) read both images and labels
3) calculate IOU for testing

'''
import os
import sys
import pandas as pd
from collections import defaultdict
import torch 
from imageio import imread, imsave
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import binarize_mask, random_crop
from pathlib import Path

from unet import UNet

IMAGE_SIZE = 256

stream_transforms = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize(size = IMAGE_SIZE), 
    transforms.ToTensor()])

def view_prediction(image, pred):
    fig = plt.figure(figsize=(20, 6), dpi=100)
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(image)
    ax1.set_title("Oncostream image")

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(pred, cmap='viridis')
    ax2.set_title("Prediction heatmap")
    plt.show(),t

def feedfoward(model, raw_image, raw_mask, transform, device = 'cuda'):
    """Function that will perform forward pass on raw unprocessed image. 
    Well then return the probability heatmap from network."""
    
    if raw_image.shape[2] != 3:
            raw_image = raw_image[:,:,0:3]

    image_crop, mask_crop = random_crop(raw_image, raw_mask)
    height, width = image_crop.shape[0], image_crop.shape[1]

    # transform image
    image = transform(image_crop)
    pred = model(image.unsqueeze(dim = 0).to(device))
    pred = pred.squeeze().cpu()
    pred = torch.ge(torch.sigmoid(pred), 0.5).float().numpy()

    # resize the prediction 
    pred = resize(pred, output_shape=(height, width), order = 3) # resize prediction
    
    return image_crop, pred, mask_crop

def calculate_area_oncostream(pred, prop_threshold = 0.9):
    """Will calculate the area of oncostream in prediction"""
    pred[pred > prop_threshold] = 1
    pred[pred <= prop_threshold] = 0
    onco_area = pred.sum()
    print(onco_area/(pred.shape[-1] ** 2))
    return onco_area/(pred.shape[-1] ** 2)


def save_predictions(model, transform, root, raw_dir, save_dir, img_filename, calculate_area = True, device = 'cpu'):
    """Function call to save predictions from a model. root_save must contain preds and crops subdirectories."""
    img = imread(os.path.join(root, img_filename))
    
    # perform feedforward pass on image
    img, pred = feedfoward(model, img, transform, device)
    if calculate_area:
        area = calculate_area_oncostream(pred)
    
    pred *= 255
    # save both the prediction and the random crop
    preds_path = os.path.join(root.replace(raw_dir, save_dir), 'preds')
    crops_path = os.path.join(root.replace(raw_dir, save_dir), 'crops')
    os.makedirs(preds_path, exist_ok=True)
    os.makedirs(crops_path, exist_ok=True)

    plt.imsave(os.path.join(preds_path, img_filename[0:-4] + "_pred.png"), pred, cmap = "viridis", vmin = 0, vmax = 255)
    plt.imsave(os.path.join(crops_path, img_filename[0:-4] + "_crop.png"), img, vmin = 0, vmax = 255)

    return area

def compute_metrics(model, 
                    image_path, 
                    mask_path,
                    transform, 
                    save_dir, 
                    calculate_area = True,
                    threshold = 0.5,
                    device = 'cpu'):
    
    image = imread(image_path)
    mask = imread(mask_path)
    mask = binarize_mask(mask)

    # perform feedforward pass on image
    img, pred, mask = feedfoward(model, image, mask, transform, device)
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    # computer area
    if calculate_area:
        area = calculate_area_oncostream(pred)
        print('area -> {}'.format(area))
    
    # computer IOU
    intersection = np.sum(mask * pred) 
    union = (np.sum(mask) + np.sum(pred)) - intersection
    iou = intersection/(union + 1e-6)
    print('iou -> {}'.format(iou))

    pred *= 255
    mask *= 255

    if 'tif' in str(image_path):
        image_path.rename(str(image_path).replace('tif', 'png'))

    if 'tif' in str(mask_path):
        mask_path.rename(str(mask_path).replace('tif', 'png'))

    final_image = image_path.parts[-4:]
    final_mask = mask_path.parts[-4:]

    final_image_path = save_dir / final_image[0] / final_image[1] / final_image[2]
    final_mask_path = save_dir / final_mask[0] / final_mask[1] / final_mask[2] 
    final_preds_path = save_dir / final_mask[0] / final_mask[1] / 'preds'

    os.makedirs(final_image_path, exist_ok=True)
    os.makedirs(final_mask_path, exist_ok=True)
    os.makedirs(final_preds_path, exist_ok=True)
   
    print(final_image_path / final_image[3])

    plt.imsave(final_image_path / final_image[3], img, vmin = 0, vmax = 255)
    plt.imsave(final_mask_path /final_mask[3], mask, cmap = "viridis", vmin = 0, vmax = 255)
    plt.imsave(final_preds_path / final_mask[3], pred, cmap = "viridis", vmin = 0, vmax = 255)
    
    return iou, area


def view_image_mask(image, mask):
	image = image.numpy()
	mask = mask.numpy()
	img = np.moveaxis(image, 0, -1)
	mask = np.moveaxis(mask, 0, -1)

	fig = plt.figure()
	ax1 = fig.add_subplot(1,2,1)
	ax1.imshow(img)
	ax1.set_title("Oncostream image")

	ax2 = fig.add_subplot(1,2,2)
	ax2.imshow(mask, cmap='viridis')
	ax2.set_title("mask")
	plt.show()


if __name__ == '__main__':

    # load model
    model = torch.load('models/unet_oncostreams_acc86.pt', map_location = torch.device('cpu'))
    model.eval()
    print(model)

    # specify directories 
    img_root = Path('/home/toddhollon/Desktop/DeepStreams/data/tcga_grade4/')
    images_path = img_root / 'images'
    masks_path = img_root / 'masks'
   
    save_dir = Path('/home/toddhollon/Desktop/DeepStreams/results')
   
    images = sorted(os.listdir(images_path))
    masks = sorted(os.listdir(masks_path))
    data = list(zip(images, masks))
   
    results_dict = defaultdict(list)
    for image, mask in data:
        image_path = images_path / image
        mask_path = masks_path / mask
        iou, area = compute_metrics(model, image_path, mask_path, stream_transforms, save_dir)
        results_dict['images'].append(str(image_path))
        results_dict['iou'].append(iou)
        results_dict['area'].append(area)
    
    df = pd.DataFrame(results_dict)
    df.to_excel(save_dir / 'model_spreadsheet.xlsx')
