import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras import backend as K
import os
import json
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense , Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D , TimeDistributed
from tensorflow.keras.layers import MaxPooling2D , ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import resnet
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
#from imutils import paths
from os import listdir
from os.path import isfile, join
#import cv2
import PIL
from itertools import combinations

wandb.init(project="SiameseNetwork", entity="aviroshkovan")
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 4 , 'CPU': 16} ) 
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)


BATCH_SIZE = 128
EPOCHS = 2
IMG_SHAPE = (35, 55, 1)
margin = 1  # Margin for constrastive loss.
data = []
labels_images1 = []
labels_images2 = []
labels = []
BASE_OUTPUT = '/mnt/md0/mory/SimGan/'
PLOT_PATH = os.path.sep.join([BASE_OUTPUT,
	"contrastive_plot2.png"])
#Load data
print("Loading Images...")

#Load synth and real images
synth_images_path= '/mnt/md0/mory/EyeGaze/synth/pre-process/mixed/'
synth_data_paths = [i for i in (os.path.join(synth_images_path, f) for f in os.listdir(synth_images_path)) if os.path.isfile(i)]
real_images_path= '/mnt/md0/mory/EyeGaze/real/pre-process/mixed/'
real_data_paths = [i for i in (os.path.join(real_images_path, f) for f in os.listdir(real_images_path)) if os.path.isfile(i)]

'''
def build_siamese_model(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)
	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)
	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)
	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)
	# build the model
	model = Model(inputs, outputs)
	# return the model to the calling function
	return model
'''
def build_siamese_model(inputShape, embeddingDim=4096):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)
	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(32, (11, 11), padding="same", kernel_regularizer=l2(2e-4), activation="relu")(inputs)
	x = BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9)(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = ZeroPadding2D((2, 2))(x)
	x = Dropout(0.3)(x)
	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (5, 5), padding="same",kernel_regularizer=l2(2e-4), activation="relu")(x)
	x = BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9)(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = Dropout(0.3)(x)

	x = Conv2D(128, (3, 3), padding="same",kernel_regularizer=l2(2e-4), activation="relu")(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = Dropout(0.3)(x)

	x = Conv2D(256, (3, 3), padding="same",kernel_regularizer=l2(2e-4), activation="relu")(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = Dropout(0.5)(x)

	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(pooledOutput)
	# build the model
	model = Model(inputs, outputs)
	# return the model to the calling function
	return model



def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA][0]
		#print(label)
		#print(idx)
		#print(idx[label])
		# randomly pick an image that belongs to the *same* class
		# label
		
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]
		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])
		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]
		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])
	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss

for imagePath in synth_data_paths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-1]
	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(35, 55), color_mode="grayscale")
	image = img_to_array(image)
	image = preprocess_input(image)
	# update the data and labels lists, respectively
	data.append(image)
	labels_images1.append(label)
# convert the data and labels to NumPy arrays
data_synth = np.array(data, dtype="float32")
#print(data)
labels_images_synth = np.array(labels_images1)
#print(labels_images_synth)
#print(data.shape)
print("synth images Loaded successfully.")

for imagePath in real_data_paths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-1]
	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(35, 55), color_mode="grayscale")
	image = img_to_array(image)
	image = preprocess_input(image)
	# update the data and labels lists, respectively
	data.append(image)
	labels_images2.append(label)
# convert the data and labels to NumPy arrays
data_real = np.array(data, dtype="float32")
#print(data)
labels_images_real = np.array(labels_images2)
#print(labels_images_real)
#print(data.shape)
print("real images Loaded successfully.")

new_images_real = data_real[:5000]
new_images_synth = data_synth[:5000]

label1 = np.ones( (new_images_real.shape[0],1) )
label1 = label1.astype(int)
label2 = np.zeros( (new_images_synth.shape[0],1) )
label2 = label2.astype(int)
data_all = np.concatenate([new_images_real, new_images_synth])
data_all = np.array(data_all)
labels_all = np.concatenate([label1, label2])
labels_all = np.array(labels_all)
#print(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

(trainX, testX, trainY, testY) = train_test_split(data_all, labels_all,
	test_size=0.20, random_state=42, shuffle = True)


trainX = trainX / 255.0
testX = testX / 255.0


# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)
print("Done making pairs!")

# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
# finally, construct the siamese network
distance = Lambda(euclidean_distance)([featsA, featsB])
model = Model(inputs=[imgA, imgB], outputs=distance)

# compile the model
print("[INFO] compiling model...")
rms = keras.optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss=contrastive_loss, optimizer=rms, run_eagerly=True)
# train the model
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	callbacks=[WandbCallback()])

print("Loading images...")

image1 = tf.keras.preprocessing.image.load_img('/mnt/md0/mory/EyeGaze/synth/pre-process/mixed/44_image.jpg', color_mode = 'grayscale', target_size=(35,55))
image1 = tf.keras.preprocessing.image.img_to_array(image1)
image2 = tf.keras.preprocessing.image.load_img('/mnt/md0/mory/EyeGaze/synth/pre-process/mixed/50_image.jpg', color_mode = 'grayscale', target_size=(35,55))
image2 = tf.keras.preprocessing.image.img_to_array(image2)
image1 = np.expand_dims(image1, axis=-1)
image2 = np.expand_dims(image2, axis=-1)
# add a batch dimension to both images
image1 = np.expand_dims(image1, axis=0)
image2 = np.expand_dims(image2, axis=0)
image1 = image1 / 255.0
image2 = image2 / 255.0

print("images loaded!")
print(image1.shape)
print(image2.shape)
preds = model.predict([image1, image2])
proba = preds[0][0]
print(proba)
# serialize the model to disk


print("[INFO] saving siamese model...")

#print("saving siamese model...")

# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('test2_model_siam.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('test2_siam_weights.h5')

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)	

#save_model(model, history, 'siam')	
plot_training(history, PLOT_PATH)
