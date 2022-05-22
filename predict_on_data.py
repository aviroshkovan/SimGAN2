# import the necessary packages
from re import A
from PIL import Image
import PIL
from numpy import random
from tensorflow.keras.models import load_model
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf
import random
from pathlib import Path
import matplotlib.pyplot as plt


def self_regularization_loss(y_true, y_pred):

    delta = 0.0001  # FIXME: need to figure out an appropriate value for this
    delta1 = 0.2
    delta2 = 2.0
    #pred = tf.reduce_sum(tf.abs(y_pred - y_true))
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

    # return tf.abs(pred)
    #return tf.multiply(delta, pred)
    return tf.multiply(delta1, loss)
    #return loss


path = os.path.dirname(os.path.abspath(__file__))
refined_cache_dir = os.path.join(path, 'refined_images_test_MSE_loss_5000_64_WithMomentum')
synth_cache_dir = os.path.join(path, 'synth_images_test_MSE_loss_5000_64_WithMomentum')


images_path = "/mnt/md0/mory/EyeGaze/synth/pre-process/mixed"
os.chdir('/mnt/md0/mory/EyeGaze/synth/pre-process/mixed')
synth_data = os.listdir('/mnt/md0/mory/EyeGaze/synth/pre-process/mixed')
images = []

# load all images into a list
random.seed(42)
random_images_sample = []
random_images_sample_index = []
while len(random_images_sample_index) < 5000:
    index = random.randint(0,len(synth_data) -1)
    if index not in random_images_sample_index:
        random_images_sample_index.append(index)
        random_images_sample.append(synth_data[index])

for img_name in random_images_sample:
    img_path = os.path.join(images_path, img_name)
    img = image.load_img(img_path, target_size=(35, 55), color_mode='grayscale')
    img_arr = image.img_to_array(img)
    #processed_image = np.array(img_arr) / 255.0
    images.append(img_arr)
   
def save_refined_images(images):

    for i, image in enumerate(images):

        output_path = str(refined_cache_dir) + '/' + f"refined_image_{i}.png"
        img = np.uint8((np.array(image[:, :, 0])))
        Image.fromarray((img).astype(np.uint8)).save(output_path)
        
def save_synth_images():
    i = 0
    for index in random_images_sample_index:
        img = synth_data[index]
        output_path_synth = str(synth_cache_dir) + '/' + f"synth_image_{i}.png"
        i +=1
        img_path = os.path.join(images_path, img)
        loaded_img = image.load_img(img_path, target_size=(35, 55), color_mode='grayscale')
        loaded_img.save(output_path_synth)


# load model
model = load_model('/mnt/md0/mory/SimGan/MSE_loss_5000_64_WithMomentum/refiner_model_step_4950.h5',
                   custom_objects={'self_regularization_loss': self_regularization_loss})


raw_images = np.array(images)
preprocessed_images = applications.xception.preprocess_input(raw_images)
pred_result = model.predict(preprocessed_images)
pred_result_postprocessed = np.uint8((np.array(pred_result[:,:,:])+1) * 255 / 2)
save_refined_images(pred_result_postprocessed)
save_synth_images()
print("Refined and Synth Images have been saved successfully")