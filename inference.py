import torch 
from imageio import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.image_utils import random_crop

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
	plt.show()

def feedfoward(model, raw_image, transform, device = 'cuda'):
	"""Function that will perform forward pass on raw unprocessed image. 
	Well then return the probability heatmap from network."""

	if raw_image.shape[2] != 3:
		raw_image = raw_image[:,:,0:3]

	image_crop = random_crop(raw_image)
	height, width = image_crop.shape[0], image_crop.shape[1]

	image = transform(image_crop)
	pred = model(image.unsqueeze(dim = 0).to(device))
	pred = pred.squeeze().cpu()
	pred = torch.ge(torch.sigmoid(pred), 0.5).float().numpy()

	# resize the prediction 
	pred = resize(pred, output_shape=(height, width), order = 3) # resize prediction
	
	return image_crop, pred


def calculate_area_oncostream(pred, prop_threshold = 0.9):
	"""Will calculate the area of oncostream in prediction"""
	pred[pred > prop_threshold] = 1
	pred[pred <= prop_threshold] = 0
	onco_area = pred.sum()
	return onco_area/(256*256)


def save_predictions(model, tranform, root, raw_dir, save_dir, img_filename, calculate_area = True):
	"""Function call to save predictions from a model. root_save must contain preds and crops subdirectories."""
	img = imread(os.path.join(root, img_filename))
	
	# perform feedforward pass on image
	image_crop, pred = feedfoward(model, img, transform)

	if calculate_area:
		area = calculate_area_oncostream(pred)
	
	# save both the prediction and the random crop
	preds_path = os.path.join(root.replace(raw_dir, save_dir), 'preds')
	crops_path = os.path.join(root.replace(raw_dir, save_dir), 'crops')
	os.makedirs(preds_path, exist_ok=True)
	os.makedirs(crops_path, exist_ok=True)

	plt.imsave(os.path.join(preds_path, img_filename[0:-4] + "_pred.png"), pred, cmap = "viridis", vmin = 0, vmax = 255)
	plt.imsave(os.path.join(crops_path, img_filename[0:-4] + "_crop.png"), img, vmin = 0, vmax = 255)

	return area


def export_file_area_list(tuple_list):
	files = [x[0] for x in tuple_list]
	areas = [x[1] for x in tuple_list]
	return DataFrame({"files":files, "areas":areas})

# if __name__ == "__main__":

model = torch.load('unet_oncostreams_acc86.pt')
print(model)

raw_image = imread('/home/labcomputer/Desktop/oncostreams/oncostream_all_raw/NewMASK2021andInputs/images/Image_6.tif')
image, pred = feedfoward(model, raw_image, stream_transforms)
print(image.shape)
print(pred.shape)
view_prediction(image, pred)

# img_root = '/Users/toddhollon/Desktop/DeepLearningforNPAandNPDColi'
# raw_dir = 'raw_images'
# save_dir = 'predictions_mouseCNN_larger'
# file_area = []
# # loop to run through and save predictions on images in single directory
# for root, dirs, files in os.walk(os.path.join(img_root, raw_dir)):
#     for file in files:
#         if "tif" in file or "JPG" in file or "png" in file:
#             print(os.path.join(root, file))
#             _, leaf = root.split(raw_dir)
#             area = save_predictions(model, root, raw_dir, save_dir, img_filename = file, calculate_area=True)
#             file_area.append((os.path.join(leaf, file), area))
# df = export_file_area_list(file_area)
# df.to_excel(os.path.join(img_root, 'prediction.xlsx')