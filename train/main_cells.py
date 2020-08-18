from model import *
from data import *
from skimage.io import imread
import skimage.transform as trans
from keras.models import load_model
import matplotlib.pyplot as plt

image_size = (256, 256, 3)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="reflect")


myGene = trainGenerator(batch_size = 2, 
                        num_class = 1, 
                        train_path = "/home/orringer-lab/Desktop/oncostreams/training_tiles",
                        image_folder='image',
                        mask_folder='mask', 
                        aug_dict=data_gen_args, 
                        image_color_mode = "rgb", 
                        mask_color_mode = "grayscale", 
                        target_size=image_size[0:2])
#                        save_to_dir = "/home/todd/Desktop/unet/tiled_images/output")

model = unet(input_size=image_size)
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, 
                    steps_per_epoch=1000, 
                    epochs=20, 
#                    callbacks=[model_checkpoint], 
                    verbose=1)


#testGene = testGenerator("/home/todd/Desktop/unet/tiled_images/test/", as_gray = False)
#results = model.predict_generator(testGene, steps = 4, verbose=1)
#saveResult("/home/todd/Desktop/unet/tiled_images/test", results)

model = load_model("/home/todd/Desktop/unet/unet_membrane.hdf5")

def plotting_function(image_path):
    
    img = imread(image_path)
    img = trans.resize(img,(256, 256))
    img_for_net = np.expand_dims(img, axis = 0)
    pred = model.predict(img_for_net)[0,:,:,0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow(img)
    ax1.set_title("Contrast enhanced image")
    
    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow(pred)
    ax2.set_title("prediction heatmap")
    
    ax3 = fig.add_subplot(1,4,3)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    ax3.imshow(pred)
    ax3.set_title("binarized prediction")
    fig.show()
    
    ax4 = fig.add_subplot(1,4,4)
    mask = imread(image_path.replace("image", "mask"))
    ax4.imshow(mask)
    ax4.set_title("Ground Truth")
    fig.show()
    
    
plotting_function("/home/orringer-lab/Desktop/oncostreams/training_tiles/image/Image_30924_26.tif")


