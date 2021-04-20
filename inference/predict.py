
import numpy as np
from pandas import DataFrame
import os
import sys
from skimage.io import imread
import skimage.transform as trans
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
from pathlib import Path

def random_crop(image, mask = None, min_percent = 60, max_percent = 100): # add mask to argument
	"""Input image as a numpy array. Output a random crop from that image. If mask is None, will just return crop image."""
	height = image.shape[0]
	width = image.shape[1]

	rand_factor = np.random.randint(low = min_percent, high = 100) / 100 # float between 0.4 and 1, want images no less than 40% of original
	rand_size = int(height * rand_factor) # nearly all images the smallest dimension is height
	print(rand_size)

	rand_y = np.random.randint(low = 0, high = height - rand_size)
	rand_x = np.random.randint(low = 0, high = width - rand_size)

	crop_image = image[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size]
	assert crop_image.shape[0] == crop_image.shape[1]

	if mask:
		crop_mask = mask[rand_y:rand_y + rand_size, rand_x:rand_x + rand_size]
		assert crop_image.shape[0] == crop_mask.shape[0]
		return crop_image, crop_mask

	else:
		return crop_image

def preprocessing_rescale(img):
	"""Simple rescaling function to make pixel values between 0-1"""
	if (np.max(img) > 1):
		return img / 255
	else:
		return img
	return img

def feedfoward(model, raw_image):
	"""Function that will perform forward pass on raw unprocessed image. Well then return the probability heatmap from network."""
	img_resize = trans.resize(raw_image, (256, 256))
	img_for_net = preprocessing_rescale(img_resize)
	pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0]
	return pred
	
def contour_plot(image_path, model):
	img = imread(image_path)
	pred = feedfoward(img)

	percentiles = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

	pred = np.flipud(pred) # realign the y axis

	fig, ax = plt.subplots()
	CS = ax.contour(pred, levels = percentiles)
	ax.clabel(CS, inline = 1)
	plt.show()

def iou(y_true, y_pred):

	# calculates a boolean array, then converts to float
	y_true = K.cast(K.greater_equal(y_true, 0.5), K.floatx())
	y_pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())

	# computes intersection and union
	intersection = K.sum(y_true * y_pred)
	union = K.sum(y_true) + K.sum(y_pred) - intersection
	# error handling if union is equal to 0, avoid zerodivision error
	return K.switch(K.equal(union, 0), 1.0, intersection/union)

def batch_iou(preds, y_test):
	iou_val = [] 
	for i in range(len(preds)):
		def indexer(array):
			indices = []
			for index, val in enumerate(array):
				if val:
					indices.append(index)
			return set(indices)

		pred_flat = indexer(preds[i,:,:,:].flatten())
		ground_flat = indexer(y_test[i,:,:,:].flatten())
		iou = len(pred_flat & ground_flat)/len(pred_flat | ground_flat)
		iou_val.append(iou)
	return iou_val

def calculate_area_oncostream(pred, prop_threshold = 0.9):
	"""Will calculate the area of oncostream in prediction"""
	pred[pred > prop_threshold] = 1
	pred[pred <= prop_threshold] = 0
	onco_area = pred.sum()
	return onco_area/(256*256)

def save_predictions(model, root, raw_dir, save_dir, img_filename, calculate_area = True):
	"""Function call to save predictions from a model. root_save must contain preds and crops subdirectories."""
	img = imread(os.path.join(root, img_filename))
	img = random_crop(img, min_percent = 75, max_percent = 95)
	height, width = img.shape[0], img.shape[1]

	# error handling where some of the images where more than 3 channels (?)
	if img.shape[2] != 3:
		img = img[:,:,0:3]

	# perform feedforward pass on image
	pred = feedfoward(model, img)

	if calculate_area:
		area = calculate_area_oncostream(pred)
	
	# save both the prediction and the random crop
	pred *= 255 # rescale prediction
	pred = trans.resize(pred, output_shape=(height, width), order = 3) # resize prediction

	preds_path = os.path.join(root.replace(raw_dir, save_dir), 'preds')
	crops_path = os.path.join(root.replace(raw_dir, save_dir), 'crops')
	os.makedirs(preds_path, exist_ok=True)
	os.makedirs(crops_path, exist_ok=True)

	plt.imsave(os.path.join(preds_path, img_filename[0:-4] + "_pred.png"), pred, cmap = "viridis", vmin = 0, vmax = 255)
	plt.imsave(os.path.join(crops_path, img_filename[0:-4] + "_crop.png"), img, vmin = 0, vmax = 255)

	return area
	
def plotting_function(image_path, model):

	img = imread(image_path)
	img = random_crop(img)
	pred = feedfoward(model, img)

	fig = plt.figure(figsize=(20, 6), dpi=100)
	ax1 = fig.add_subplot(1,4,1)
	diff = 1 - img.max()
	ax1.imshow(img + diff)
	ax1.set_title("Oncostream image")

	ax2 = fig.add_subplot(1,4,2)
	ax2.imshow(pred, cmap='Spectral_r')
	ax2.set_title("Prediction heatmap")

	ax3 = fig.add_subplot(1,4,3)
	pred[pred > 0.5] = 1
	pred[pred <= 0.5] = 0
	ax3.imshow(pred, cmap="Greys_r")
	ax3.set_title("Binarized prediction")

	mask = imread(image_path.replace("images", "masks"))
	mask = trans.resize(mask,(256, 256))
	ax4 = fig.add_subplot(1,4,4)
	ax4.imshow(mask)
	ax4.set_title("Ground truth")

	def indexer(array):
		indices = []
		for index, val in enumerate(array):
			if val:
				indices.append(index)
		return set(indices)

	pred_flat = indexer(pred.flatten())

	ground_flat = indexer(mask[:,:,0].flatten())
	iou = len(pred_flat & ground_flat)/len(pred_flat | ground_flat)
	plt.suptitle("Intersection over union: " + str(np.round(iou, 3)))

def plotting_function_inference(model, image_path = None, generator = None):
	"""Plot function"""
	if generator:
		# index into the input image and the first channel 
		img, mask = generator.__next__()
		img, mask = img[0], mask[0]
	if image_path:
		img = imread(image_path)
		img = random_crop(img)
	
	# error handling where some of the images where more than 3 channels (?)
	if img.shape[2] != 3:
		img = img[:,:,0:3]
	
	# forward pass through network and index into the prediction
	pred = feedfoward(model, img)	

	# img = img[0] # most reduce dims 
	fig = plt.figure()
	ax1 = fig.add_subplot(1,3,1)
	# diff = 1 - img_copy.max()
	# ax1.imshow(img_copy + diff)
	ax1.imshow(img)
	ax1.set_title("Oncostream image")

	ax2 = fig.add_subplot(1,3,2)
	ax2.imshow(pred, cmap='viridis')
	ax2.set_title("Prediction heatmap")

	ax3 = fig.add_subplot(1,3,3)
	pred[pred > 0.5] = 1
	pred[pred <= 0.5] = 0
	ax3.imshow(pred, cmap="Greys_r")
	ax3.set_title("Binarized prediction")

	plt.suptitle(image_path.split("/")[-1])
	plt.show()
	# plt.savefig("oncoprediction.png", dpi = 500)

def export_oncostream_area(model, image_dir):
	filelist = sorted(os.listdir(image_dir))
	arealist = []
	for file in filelist:
		print(file)
		arealist.append(calculate_area_oncostream(os.path.join(image_dir, file), model))
	return DataFrame({"files":filelist, "areas":arealist})

def export_file_area_list(tuple_list):
	files = [x[0] for x in tuple_list]
	areas = [x[1] for x in tuple_list]
	return DataFrame({"files":files, "areas":areas})

if __name__ == "__main__":

	# import model, must include custom_object if used intersection over union as data metric
	# model = load_model("/Users/toddhollon/Desktop/oncostreams_FINAL/models/human_oncostreams_94trainacc.hdf5")
	model = load_model("/Users/toddhollon/Desktop/oncostreams_FINAL/models/oncostreams_fcnn_97trainaccTODDBESTMODEL.hdf5")
	# model = load_model("/Users/toddhollon/Desktop/oncostreams_mac/model_trained_val1.hdf5", custom_objects = {'iou':iou})

	img_root = '/Users/toddhollon/Desktop/DeepLearningforNPAandNPDColi'
	raw_dir = 'raw_images'
	save_dir = 'predictions_mouseCNN_larger'
	file_area = []
	# loop to run through and save predictions on images in single directory
	for root, dirs, files in os.walk(os.path.join(img_root, raw_dir)):
	    for file in files:
			if "tif" in file or "JPG" in file or "png" in file:
				print(os.path.join(root, file))
				_, leaf = root.split(raw_dir)
				area = save_predictions(model, root, raw_dir, save_dir, img_filename = file, calculate_area=True)
				file_area.append((os.path.join(leaf, file), area))
	df = export_file_area_list(file_area)
	df.to_excel(os.path.join(img_root, 'prediction.xlsx'))
