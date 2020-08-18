#!/usr/bin/env python3

"""
Main script to train and analyze semantic segmentation of oncostreams in H&E images
"""

from models.unet_architectures import unet, att_unet, r2_unet, att_r2_unet
from utils.data import trainGenerator
import numpy as np
import os
from pandas import DataFrame
from pylab import rcParams

# from skimage.io import imread
from imageio import imread
import skimage.transform as trans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pandas import DataFrame

# import keras and tf
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras import backend as K

def preprocessing_rescale(img):
    '''
    Simple rescaling function to make pixel values between 0-1.
    The current preprocessing is just this rescaling with mean subtraction.
    '''
    if (np.max(img) > 1):
        return img / 255
    else:
        return img
    return img

def iou(y_true, y_pred):

    # calculates a boolean array, then converts to float
    y_true = K.cast(K.greater_equal(y_true, 0.5), K.floatx())
    y_pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())

    # computes intersection and union
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # error handling if union is equal to 0, avoid zerodivision error
    return K.switch(K.equal(union, 0), 1.0, intersection/union)

def iou_metric(y_true_path, y_pred_path):
    y_true = imread(y_true_path)
    y_pred = imread(y_pred_path)

    y_true = trans.resize(y_true, (256, 256))[:,:,0]
    y_pred = trans.resize(y_pred, (256, 256))

    # computes intersection and union
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    # error handling if union is equal to 0, avoid zerodivision error
    print(intersection/union)

def mask_path_modifiers(path):
    path = path.replace("images", "masks")
    path = path.replace("image", "mask")
    return path

def forward_pass(image_path, model):
    img = imread(image_path)

    if img.shape[2] != 3:
        img = img[:,:,0:3]

    img_resize = trans.resize(img, (256, 256))
    img_for_net = preprocessing_rescale(img_resize)
    pred = model.predict(img_for_net[None,:,:,:])[0,:,:,0]
    return pred

def export_oncostream_area(image_dir, model):
    filelist = sorted(os.listdir(image_dir))
    arealist = []
    for file in filelist:
        print(file)
        arealist.append(calculate_area_oncostream(os.path.join(image_dir, file), model))
    return DataFrame({"files":filelist, "areas":arealist})

def step_decay(epoch):
    initial_lr = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lr * np.power(drop, np.floor((epoch + 1)/epochs_drop))
    return lrate

def train_histories(epochs, validation_dir):
    """
    Returns a dataframe of the training history
    """
    model = load_model("/home/orringer-lab/Desktop/oncostreams/models/oncostreams_fcnn_97trainacc.hdf5")
    validation_generator = trainGenerator(batch_size = 4,
                        train_path = validation_dir,
                        image_folder = 'images',
                        mask_folder = 'masks',
                        aug_dict = data_gen_args,
                        save_to_dir = None)

    lrate_schedule = LearningRateScheduler(step_decay)

    model.compile(optimizer = Adam(lr = learn_rate), loss = 'binary_crossentropy', metrics = ['accuracy', iou])
    history = model.fit_generator(train_generator,
                                    steps_per_epoch=300,
                                    epochs=epochs,
                                    validation_data=validation_generator,
                                    callbacks=[lrate_schedule],
                                    validation_steps=25,
                                    shuffle = True)

    return (DataFrame(history.history), model)

if __name__ == "__main__":

    # import  
    IMAGE_SIZE = (256, 256, 3)
    # specify the root directory with "images" and "masks" subdirectories
    training_image_dir = ""
    validation_image_dir = ""

    # instantiate the data generator augmentation methods
    # this may need to change for train versus validation generator 
    data_gen_args = dict(rotation_range=0.0,
                        width_shift_range=0.0,
                        height_shift_range=0.0,
                        shear_range=0.0,
                        zoom_range=0.0,
                        vertical_flip = True,
                        horizontal_flip=True,
                        fill_mode=None)

    train_gen = trainGenerator(batch_size=5,
                            train_path= training_image_dir,
                            image_folder= 'images',
                            mask_folder= 'masks',
                            aug_dict= data_gen_args,
                            save_to_dir = None)

    val_gen = trainGenerator(batch_size=5,
                            train_path= validation_image_dir,
                            image_folder= 'images',
                            mask_folder= 'masks',
                            aug_dict= data_gen_args,
                            save_to_dir = None)

    # model = load_model('/media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home/todd/Desktop/oncostreams/python_scripts/oncostreams_cells_98trainacc.hdf5')
    # model = att_r2_unet()
    model = unet(input_size=IMAGE_SIZE)

    # compile the model
    adam = Adam(lr = 0.0005)
    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # fit the model
    model.fit_generator(train_gen, steps_per_epoch=1000, epochs=20),

    # df = pd.DataFrame(model.history.history)
    # plotting_function_infeirence("/media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home/todd/Desktop/oncostreams/patches/images/1_18.png", model)
    # plotting_function("/home/todd/Desktop/oncostreams/training_tiles/images/1_35.tif", model)
    # plotting_function_inference("/home/todd/Desktop/oncostreams/old_dirs/testing_tiles/images/Image_25250.tif", model)

    # model.save("oncostreams_cells_98trainacc.hdf5")

    # image_dir = "/Volumes/UNTITLED/TODD-2019-05-19/"
    # filelist = os.listdir(image_dir)
    # for file in filelist:
    #     plotting_function_inference(os.path.join(image_dir, file), model)

#     training_image_dir = '/home/orringer-lab/Desktop/oncostreams/random_crops'
#     validation_image_dir = "/home/orringer-lab/Desktop/oncostreams/vallsidation_set"
#     validation_dir_list = sorted(os.listdir(validation_image_dir))
#     validation_dir_list = [os.path.join(validation_image_dir, x) for x in validation_dir_list]
#     cell_dir = "/home/orringer-lab/Desktop/unet/tiled_images/train"
#     learn_rate = 0.00001

#     train_generator = trainGenerator(batch_size = 5,
#                             train_path = training_image_dir,
#                             image_folder = 'images', 
#                             mask_folder = 'masks', 
#                             aug_dict = data_gen_args, 
#                             save_to_dir = None)

#     for val_dir in validation_dir_list:
#         train_df, model = train_histories(epochs = 25, validation_dir = val_dir)
#         train_df.to_excel(val_dir.split("/")[-1] + ".xlsx")
#         model.save('model_trained_' + val_dir + ".hdf5")
