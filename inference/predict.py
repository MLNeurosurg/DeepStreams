
import numpy as np
import pandas as pd
import os
from skimage.io import imread
import skimage.transform as trans
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def random_crop(image, mask = None, min_percent = 60): # add mask to argument
	"""Input image as a numpy array. Output a random crop from that image. If mask is None, will just return crop image."""
	height = image.shape[0]
	width = image.shape[1]

	# while True:
	# select a random x and y pixel as starting point
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
    '''
    Simple rescaling function to make pixel values between 0-1
    '''
    if (np.max(img) > 1):
        return img / 255
    else:
        return img
    return img

def contour_plot(image_path, model):
    img = imread(image_path)
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0] ##### MAKE SURE CORRECT!!

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

def save_predictions(model, image_path, save_file):
    "Function call to save predictions from a model"
    img = imread(image_path)
    img = random_crop(img, min_percent = 90)
    height, width = img.shape[0], img.shape[1]

    # error handling where some of the images where more than 3 channels (?)
    if img.shape[2] != 3:
        img = img[:,:,0:3]

    # image preprocess and predicti
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0]

    # save both the prediction and the random crop
    pred *= 255
    pred = trans.resize(pred, output_shape=(height, width), order = 3)
    plt.imsave(image_path[0:-4] + "_pred.png", pred, cmap = "viridis", vmin = 0, vmax = 255)
    plt.imsave(image_path[0:-4] + "_crop.png", img, vmin = 0, vmax = 255)

def calculate_area_oncostream(image_path, model):

    img = imread(image_path)
    if img.shape[2] != 3:
        img = img[:,:,0:3]
    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0] ##### MAKE SURE CORRECT!!

    pred[pred > 0.90] = 1
    pred[pred <= 0.90] = 0
    onco_area = pred.sum()
    return onco_area/(256*256)

def plotting_function(image_path, model):

    img = imread(image_path)
    img_resize = trans.resize(img,(256, 256))
    img_for_net = np.expand_dims(img_resize, axis = 0)
    pred = model.predict(preprocessing(img_for_net))[0,:,:,0]

    fig = plt.figure(figsize=(20, 6), dpi=100)
    ax1 = fig.add_subplot(1,4,1)
    diff = 1 - img_resize.max()
    ax1.imshow(img_resize + diff)
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

    if generator:
        # index into the input image and the first channel 
        img, mask = generator.__next__()
        img, mask = img[0], mask[0]
    if image_path:
        img = imread(image_path)
        img = trans.resize(img,(256, 256))
        img = preprocessing(img)
    
    # forward pass through network and index into the prediction
    img = np.expand_dims(img, axis = 0)
    pred = model.predict(img)[0,:,:,0]

    img = img[0] # most reduce dims 
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    diff = 1 - img.max()
    ax1.imshow(img + diff)
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
#    plt.show()
    plt.savefig("oncoprediction.png", dpi = 500)


def export_oncostream_area(image_dir, model):
    filelist = sorted(os.listdir(image_dir))
    arealist = []
    for file in filelist:
        print(file)
        arealist.append(calculate_area_oncostream(os.path.join(image_dir, file), model))
    return DataFrame({"files":filelist, "areas":arealist})

if __name__ == "__main__":

    # import model, must include custom_object if used intersection over union as data metric
    model = load_model("/Users/toddhollon/Desktop/oncostreams_mac/oncostreams_fcnn_97trainaccTODDBESTMODEL.hdf5", custom_objects = {'iou':iou})

    # call plotting functions above
    plotting_function_inference(model, image_path = "/media/4tbhd/home/todd/Desktop/oncostreams/ImagesforDeeplearningAnalysisofGBMandLGG/GBM_IV/Page1_10_DX1.JPG")    

    # can calculate the area of calculated oncostreams for specific image directory
    df = export_oncostream_area("/Users/toddhollon/Desktop/HEimages_for_deeplearning_anlaysis/stream_images", model)
    df.to_excel("andrea_data.xlsx")

    # loop to run through and save predictions on images in single directory
    for root, dirs, files in os.walk("/Users/toddhollon/Desktop/Images_for_Deep_Learning_TCGA/crops"):
        for file in files:
            if "tif" in file or "JPG" in file or "png" in file:
                print(os.path.join(root, file))
                save_predictions(os.path.join(root, file), file, model=model)
/media/4tbhd/home/todd/Desktop/oncostreams/ImagesforDeeplearningAnalysisofGBMandLGG/GBM_IV