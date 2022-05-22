"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""

import os
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
#from dlutils import plot_image_batch_w_labels
#import plot_image_batch_w_labels
import argparse
from utils.image_history_buffer import ImageHistoryBuffer
import wandb
from wandb.keras import WandbCallback
import keras.backend as K
#from keras.constraints import Constraint
from PIL import Image
import matplotlib
import numpy as np

matplotlib.use('Agg')  # b/c matplotlib is such a great piece of software ;) - needed to work on ubuntu
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'axes.titlesize': 1})

wandb.init(project="SimGAN2", entity="aviroshkovan")
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 4 , 'CPU': 16} ) 
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
#
# directories
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'Siamese_loss_5000_64_Delta2_Momentum')
images_cache_dir = os.path.join(path, 'Siamese_Images_5000_64_Delta2_Momentum')

#
# image dimensions
#

img_width = 55
img_height = 35
img_channels = 1

#
# training params
#

nb_steps = 5000
batch_size = 64
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 50


def save_image(images):
    
    num_images = len(images)
    for i, image in enumerate(images):

        if i <= 63:
            output_path = str(images_cache_dir) + '/' + f"syntethic_image_{i}.png"
        else:
            output_path = str(images_cache_dir) + '/' + f"refined_image_{i}.png"
        
        img = np.uint8((np.array(image[:,:,0])+1) * 255 / 2)
        Image.fromarray((img).astype(np.uint8)).save(output_path)

def plot_batch(image_batch, figure_path, label_batch=None, vmin=0, vmax=255, scale=True):
    """
    Plots a batch of images and their corresponding label(s)/annotations, saving the plot to disc.
    :param image_batch: Batch of images to be plotted.
    :param figure_path: Full path of the filename the plot will be saved as.
    :param label_batch: Batch of labels corresponding to `image_batch`.
       Labels will be displayed along w/ their corresponding image.
    """
    if label_batch is not None:
        assert len(image_batch) == len(label_batch), 'Their must be a label for each image to be plotted.'

    batch_size = len(image_batch)
    assert batch_size >= 1

    assert isinstance(image_batch, np.ndarray), 'image_batch must be an np array.'

    # for gray scale images if image_batch.shape == (img_height, img_width, 1) plt requires this to be reshaped
    if image_batch.shape[-1] == 1:
        image_batch = np.reshape(image_batch, newshape=image_batch.shape[:-1])

    # plot images in rows and columns
    # `+ 2` prevents plt.subplots from throwing: `TypeError: 'AxesSubplot' object does not support indexing` when batch_size < 10
    nb_rows = batch_size // 10 + 2  # each row will have 10 images, last row will have the rest of images in the batch
    nb_columns = 10
    
    _, ax = plt.subplots(nb_rows, nb_columns, sharex=True, sharey=True)

    for i in range(nb_rows):
        for j in range(nb_columns):
            try:
                x = image_batch[i * nb_columns + j]
                if scale:
                    x = x + max(-np.min(x), 0)
                    x_max = np.max(x)
                    if x_max != 0:
                        x /= x_max
                    x *= 255

                ax[i][j].imshow(x.astype('uint8'), vmin=vmin, vmax=vmax, interpolation='lanczos' )
                if label_batch is not None:
                    ax[i][j].set_title(label_batch[i * nb_columns + j])
                ax[i][j].set_axis_off()
            except IndexError:
                break

    plt.savefig(os.path.join(figure_path), dpi=600)
    plt.close()

# define a function to load existing models
def load_model(model, weights=None):
    """ Load a given TF model and it's corresponding weights,
    can be a json + h5 file, or a path to a pb file"""

    if model.endswith('.json'):
        assert weights, 'Weights File was not Provided!'
        # load json and create model
        json_file = open('test1_model_siam.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("test1_siam_weights.h5")
    else:
        # load a model from a pb file
        loaded_model = models.load_model(model)
    print('Loaded model from disk')

    return loaded_model


def refiner_network(input_image_tensor):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.

    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = layers.Convolution2D(
            nb_features, nb_kernel_rows, padding='same')(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(
            nb_features, nb_kernel_rows, padding='same')(y)

        y = layers.Add()([input_features, y])
        return layers.Activation('relu')(y)

    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, 3, padding='same',
                             activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for _ in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 × 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    return layers.Convolution2D(img_channels, 1, padding='same', activation='tanh')(x)


def discriminator_network(input_image_tensor):
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    x = layers.Convolution2D(96, 3, padding='same', strides=(
        2, 2),  activation='relu')(input_image_tensor)
    x = layers.Convolution2D(64, 3, padding='same',
                             strides=(2, 2),  activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(
        3, 3), padding='same', strides=(1, 1))(x)
    x = layers.Convolution2D(32, 3, padding='same',
                             strides=(1, 1),  activation='relu')(x)
    x = layers.Convolution2D(32, 1, padding='same',
                             strides=(1, 1),  activation='relu')(x)
    x = layers.Convolution2D(2, 1, padding='same',
                             strides=(1, 1),  activation='relu')(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)


def adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path=None, discriminator_model_path=None,
                         simnet_model_path=None, simnet_weights_path=None):
    """Adversarial training of refiner network Rθ and discriminator network Dφ."""
    #
    # define model input and output tensors
    #

    synthetic_image_tensor = layers.Input(
        shape=(img_height, img_width, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(
        shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    #
    # define models
    #

    refiner_model = models.Model(
        inputs=synthetic_image_tensor, outputs=refined_image_tensor, name='refiner')
    discriminator_model = models.Model(inputs=refined_or_real_image_tensor, outputs=discriminator_output,
                                       name='discriminator')

    # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
    refiner_model_output = refiner_model(synthetic_image_tensor)
    combined_output = discriminator_model(refiner_model_output)
    combined_model = models.Model(inputs=synthetic_image_tensor, outputs=[refiner_model_output, combined_output],
                                  name='combined')

    discriminator_model_output_shape = discriminator_model.output_shape

    print(refiner_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    #
    # define custom l1 loss function for the refiner
    #

    # load model
    simnet_model = None
    if simnet_model_path is not None:
        simnet_model = load_model(simnet_model_path, simnet_weights_path)
        print('Loaded similarity model successfully')
    '''
    def resize_tensor(tensor):
        tensor = tf.repeat(tensor, 3, axis=3)
        tensor_list = []
        unstack_y_true = tf.unstack(tensor, axis=0)
        for slice in unstack_y_true:
            tensor_list.append(tf.image.resize(slice, (32, 32)))
        tensor = tf.stack(tensor_list, axis=0)
    
        return tensor
    '''
    def self_regularization_loss(y_true, y_pred):

        delta = 0.0001  # FIXME: need to figure out an appropriate value for this
        delta1 = 0.2
        delta2 = 2.0
        delta3 = 3.0
        if simnet_model is not None:
            #if y_true.shape[-1] == 1:
            #    y_true = resize_tensor(y_true)
            #if y_pred.shape[-1] == 1:
            #    y_pred = resize_tensor(y_pred)
            pred = simnet_model([y_pred, y_true])
        else:
            #pred = tf.reduce_sum(tf.abs(y_pred - y_true))
            #pred = K.mean(y_true * y_pred)
            #margin = 1
            #square_pred = tf.math.square(y_pred)
            #margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
            loss = tf.keras.losses.mean_squared_error(y_true, y_pred)


        #return tf.math.reduce_mean(tf.abs((1 - y_true) * square_pred + (y_true) * margin_square))
        #return tf.abs(pred)   
        return tf.multiply(delta, pred)
        #return tf.multiply(delta2, loss)
        #return loss
        

    #
    # define custom local adversarial loss (softmax for each image section) for the discriminator
    # the adversarial loss function is the sum of the cross-entropy losses over the local patches
    #

    def local_adversarial_loss(y_true, y_pred):
        # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
        # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)

        return tf.reduce_mean(loss)

    #
    # compile models
    #

    sgd = optimizers.SGD(learning_rate=0.001, momentum=0.8)
    #sgd = optimizers.SGD(learning_rate=0.001)

    refiner_model.compile(optimizer=sgd, loss=self_regularization_loss)
    discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss)
    #discriminator_model.trainable = False
    combined_model.compile(optimizer=sgd, loss=[
                           self_regularization_loss, local_adversarial_loss])

    #
    # data generators
    #

    datagen = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input)

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}

    synthetic_generator = datagen.flow_from_directory(
        directory=synthesis_eyes_dir,
        **flow_from_directory_params
    )

    real_generator = datagen.flow_from_directory(
        directory=mpii_gaze_dir,
        **flow_from_directory_params
    )

    def get_image_batch(generator):
        """tensorflow.keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch

    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    y_real = np.array(
        [[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
    y_refined = np.array(
        [[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
    assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)

    if not refiner_model_path:
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network...')
        shape = len(refiner_model.metrics_names)
        if not refiner_model.metrics_names:
            shape = 1
        gen_loss = np.zeros(shape=shape)

        for i in range(1000):
            synthetic_image_batch = get_image_batch(synthetic_generator)
            gen_loss = np.add(refiner_model.train_on_batch(
                synthetic_image_batch, synthetic_image_batch), gen_loss)

            # log every `log_interval` steps
            if not i % log_interval:
                figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(
                    i)
                print(
                    'Saving batch of refined images during pre-training at step: {}.'.format(i))

                synthetic_image_batch = get_image_batch(synthetic_generator)
                plot_batch(
                    np.concatenate(
                        (synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch))),
                    os.path.join(cache_dir, figure_name),
                    label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

                print('Refiner model self regularization loss: {}.'.format(
                    gen_loss / log_interval))
                gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        refiner_model.save(os.path.join(
            cache_dir, 'refiner_model_pre_trained.h5'))
    else:
        refiner_model.load_weights(refiner_model_path)

    if not discriminator_model_path:
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network...')
        shape = len(discriminator_model.metrics_names)
        if not discriminator_model.metrics_names:
            shape = 1
        disc_loss = np.zeros(shape=shape)

        for _ in range(100):
            real_image_batch = get_image_batch(real_generator)
            disc_loss = np.add(discriminator_model.train_on_batch(
                real_image_batch, y_real), disc_loss)

            synthetic_image_batch = get_image_batch(synthetic_generator)
            refined_image_batch = refiner_model.predict_on_batch(
                synthetic_image_batch)
            disc_loss = np.add(discriminator_model.train_on_batch(
                refined_image_batch, y_refined), disc_loss)

        discriminator_model.save(os.path.join(
            cache_dir, 'discriminator_model_pre_trained.h5'))

        # hard-coded for now
        print('Discriminator model loss: {}.'.format(disc_loss / (100 * 2)))
    else:
        discriminator_model.load_weights(discriminator_model_path)

    # TODO: what is an appropriate size for the image history buffer?
    image_history_buffer = ImageHistoryBuffer(
        (0, img_height, img_width, img_channels), batch_size * 100, batch_size)

    shape_comb = len(combined_model.metrics_names)
    shape_disc = len(discriminator_model.metrics_names)
    if not combined_model.metrics_names:
        shape_comb = 1
    if not discriminator_model.metrics_names:
        shape_disc = 1
    combined_loss = np.zeros(shape=shape_comb)
    disc_loss_real = np.zeros(shape=shape_disc)
    disc_loss_refined = np.zeros(shape=shape_disc)

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the refiner
        for _ in range(k_g * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,
                                                                 [synthetic_image_batch, y_real]), combined_loss)

        for _ in range(k_d):
            # sample a mini-batch of synthetic and real images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            real_image_batch = get_image_batch(real_generator)

            # refine the synthetic images w/ the current refiner
            refined_image_batch = refiner_model.predict_on_batch(
                synthetic_image_batch)

            # use a history of refined images
            half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
            image_history_buffer.add_to_image_history_buffer(
                refined_image_batch)

            if len(half_batch_from_image_history):
                refined_image_batch[:batch_size //
                                    2] = half_batch_from_image_history

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(
                real_image_batch, y_real), disc_loss_real)
            disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                       disc_loss_refined)

        if not i % log_interval:
            # plot batch of refined images w/ current refiner
            figure_name = 'refined_image_batch_step_{}.png'.format(i)
            print('Saving batch of refined images at adversarial step: {}.'.format(i))

            synthetic_image_batch = get_image_batch(synthetic_generator)
            synthetic_and_refined_images = np.concatenate((synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch)))
            print(synthetic_and_refined_images.shape)
            plot_batch(
                synthetic_and_refined_images,
                os.path.join(cache_dir, figure_name),
                label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)
            save_image(synthetic_and_refined_images)    

            # log loss summary
            print('Refiner model loss: {}.'.format(
                combined_loss[0] / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(
                disc_loss_real / (log_interval * k_d * 2)))
            print('Discriminator model loss refined: {}.'.format(
                disc_loss_refined / (log_interval * k_d * 2)))

            wandb.log({'Discriminator_loss_real': np.mean(disc_loss_real / (log_interval * k_d * 2)),
                       'Discriminator_loss_refined': np.mean(disc_loss_refined / (log_interval * k_d * 2)),
                       'Refiner model loss': np.mean(combined_loss[0] / (log_interval * k_g * 2))
                       })

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(
                shape=len(discriminator_model.metrics_names))
            disc_loss_refined = np.zeros(
                shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(
                cache_dir, '{}_model_step_{}.h5')
            refiner_model.save(model_checkpoint_base_name.format('refiner', i))
            discriminator_model.save(
                model_checkpoint_base_name.format('discriminator', i))


def main(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path,
         simnet_model_path, simnet_weights_path):
    adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path,
                         simnet_model_path, simnet_weights_path)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth_dataset_path',
                        help='path to synthetic dataset', required=True)
    parser.add_argument('--real_dataset_path',
                        help='path to real dataset', required=True)
    parser.add_argument('--refiner_model_path',
                        help='path to refiner model', default=None)
    parser.add_argument('--discriminator_model_path',
                        help='path to refiner model', default=None)
    parser.add_argument('--simnet_model_path',
                        help='path to simnet model', default=None)
    parser.add_argument('--simnet_weights_path',
                        help='path to simnet weights', default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = setup()

if __name__ == '__main__':
    # TODO: if pre-trained models are passed in, we don't take the steps they've been trained for into account
    args = setup()
    main(args.synth_dataset_path, args.real_dataset_path, args.refiner_model_path, args.discriminator_model_path,
         args.simnet_model_path, args.simnet_weights_path)
