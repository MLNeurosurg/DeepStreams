
import numpy as np
import pandas as pd
import os
from skimage.io import imread
import skimage.transform as trans
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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

def save_predictions(image_path, save_file, model):

    img = imread(image_path)

    height, width = img.shape[0], img.shape[1]

    if img.shape[2] != 3:
        img = img[:,:,0:3]

    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0]

    # save predictions as both greyscale and viridis colormap
    pred *= 255
    pred = trans.resize(pred, output_shape=(height, width))
    plt.imsave(image_path[0:-4] + "_pred.png", pred, cmap = "viridis", vmin = 0, vmax = 255)

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

def plotting_function_inference(image_path, model):

    img = imread(image_path)
    img_resize = trans.resize(img,(256, 256))
    img_for_net = np.expand_dims(img_resize, axis = 0)
    pred = model.predict(preprocessing(img_for_net))[0,:,:,0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    diff = 1 - img_resize.max()
    ax1.imshow(img_resize + diff)
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
    plotting_function_inference("/Users/toddhollon/Desktop/Images_for_Deep_Learning_TCGA/15-Page5_1_DX8.JPG", model)    
    plotting_function_inference("/Users/toddhollon/Desktop/Images_for_Deep_Learning_TCGA/crops/18-Page3_TCGA-27-2526_3-1.tif", model)    

    # can calculate the area of calculated oncostreams for specific image directory
    df = export_oncostream_area("/Users/toddhollon/Desktop/HEimages_for_deeplearning_anlaysis/stream_images", model)
    df.to_excel("andrea_data.xlsx")

    # loop to run through and save predictions on images in single directory
    for root, dirs, files in os.walk("/Users/toddhollon/Desktop/Images_for_Deep_Learning_TCGA/crops"):
        for file in files:
            if "tif" in file or "JPG" in file or "png" in file:
                print(os.path.join(root, file))
                save_predictions(os.path.join(root, file), file, model=model)
