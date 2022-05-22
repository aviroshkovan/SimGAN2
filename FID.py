# example of calculating the frechet inception distance in Keras
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.xception  import Xception 
from os import listdir
from os.path import isfile, join
import numpy
from numpy.random import shuffle
import cv2
from skimage.transform import resize


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

#Load real images
real_images_path='/mnt/md0/mory/EyeGaze/real/pre-process/mixed'
onlyfiles = [ i for i in listdir(real_images_path) if isfile(join(real_images_path,i)) ]
images_real = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images_real[n] = cv2.imread( join(real_images_path,onlyfiles[n]), cv2.IMREAD_GRAYSCALE )
  images_real[n] = np.repeat(images_real[n][:,:, np.newaxis], 3, -1)
  #print(images_real[n].shape)

#Load refined images
refined_images_path = 'refined_images_test_MSE_loss_5000_64_Delta2_Momentum'
onlyfiles1 = [ f for f in listdir(refined_images_path) if isfile(join(refined_images_path,f)) ]
images_refined = numpy.empty(len(onlyfiles1), dtype=object)
for m in range(0, len(onlyfiles1)):	
  images_refined[m] = cv2.imread( join(refined_images_path,onlyfiles1[m]), cv2.IMREAD_GRAYSCALE )
  images_refined[m] = np.repeat(images_refined[m][:,:, np.newaxis], 3, -1)
  #print(images_refined[m].shape)

#shuffle(images_real)
images1 = images_real[:5000]
#shuffle(images_refined)
images2 = images_refined[:5000]

# convert integer to floating point values
images1 = images1.astype('object')
images2 = images2.astype('object')
print(images1[1].shape)
print(images2[1].shape)

# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)

# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

# fid between images1 and images2
fid = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid)

