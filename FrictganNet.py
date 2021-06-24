'''
Image-to-Friction implementation using Tensorflow for paper
GAN-based Image-to-Friction Generation for Tactile Simulation of Fabric Material
Author Shaoyu Cai
'''

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio
import tensorflow.keras as keras
import os
from matplotlib import pyplot as plt
from IPython import display
import cv2
import librosa.display
from scipy import signal
import argparse
import datetime
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


log_dir = "./logs/"
PATH = "./dataset/"
summary_writer = tf.summary.create_file_writer(log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# 10 kinds of materials
# Training
f1 = [os.path.join(PATH, 'velvet/5/visual/', "%d.jpg" % i) for i in range(0, 480)]
f2 = [os.path.join(PATH, 'cotton/27/visual/', "%d.jpg" % i) for i in range(0, 480)]
f3 = [os.path.join(PATH, 'leather/32/visual/', "%d.jpg" % i) for i in range(0, 480)]
f4 = [os.path.join(PATH, 'fiber/45/visual/', "%d.jpg" % i) for i in range(0, 480)]
f5 = [os.path.join(PATH, 'chiffon/57/visual/', "%d.jpg" % i) for i in range(0, 480)]
f6 = [os.path.join(PATH, 'wool/63/visual/', "%d.jpg" % i) for i in range(0, 480)]
f7 = [os.path.join(PATH, 'nylon/78/visual/', "%d.jpg" % i) for i in range(0, 480)]
f8 = [os.path.join(PATH, 'polyester/97/visual/', "%d.jpg" % i) for i in range(0, 480)]
f9 = [os.path.join(PATH, 'linen/103/visual/', "%d.jpg" % i) for i in range(0, 480)]
f10 = [os.path.join(PATH, 'silk/111/visual/', "%d.jpg" % i) for i in range(0, 480)]
filepaths = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10

f_1 = [os.path.join(PATH, 'velvet/5/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_2 = [os.path.join(PATH, 'cotton/27/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_3 = [os.path.join(PATH, 'leather/32/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_4 = [os.path.join(PATH, 'fiber/45/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_5 = [os.path.join(PATH, 'chiffon/57/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_6 = [os.path.join(PATH, 'wool/63/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_7 = [os.path.join(PATH, 'nylon/78/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_8 = [os.path.join(PATH, 'polyester/97/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_9 = [os.path.join(PATH, 'linen/103/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
f_10 = [os.path.join(PATH, 'silk/111/tactile/spectrogram/', "%d.npy" % i) for i in range(0, 480)]
filepaths_1 = f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10


# testing & validing
p1 = [os.path.join(PATH, 'velvet/5/visual/', "%d.jpg" % i) for i in range(480, 540)]
p2 = [os.path.join(PATH, 'cotton/27/visual/', "%d.jpg" % i) for i in range(480, 540)]
p3 = [os.path.join(PATH, 'leather/32/visual/', "%d.jpg" % i) for i in range(480, 540)]
p4 = [os.path.join(PATH, 'fiber/45/visual/', "%d.jpg" % i) for i in range(480, 540)]
p5 = [os.path.join(PATH, 'chiffon/57/visual/', "%d.jpg" % i) for i in range(480, 540)]
p6 = [os.path.join(PATH, 'wool/63/visual/', "%d.jpg" % i) for i in range(480, 540)]
p7 = [os.path.join(PATH, 'nylon/78/visual/', "%d.jpg" % i) for i in range(480, 540)]
p8 = [os.path.join(PATH, 'polyester/97/visual/', "%d.jpg" % i) for i in range(480, 540)]
p9 = [os.path.join(PATH, 'linen/103/visual/', "%d.jpg" % i) for i in range(480, 540)]
p10 = [os.path.join(PATH, 'silk/111/visual/', "%d.jpg" % i) for i in range(480, 540)]

Afilepaths = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10


p_1 = [os.path.join(PATH, 'velvet/5/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_2 = [os.path.join(PATH, 'cotton/27/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_3 = [os.path.join(PATH, 'leather/32/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_4 = [os.path.join(PATH, 'fiber/45/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_5 = [os.path.join(PATH, 'chiffon/57/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_6 = [os.path.join(PATH, 'wool/63/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_7 = [os.path.join(PATH, 'nylon/78/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_8 = [os.path.join(PATH, 'polyester/97/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_9 = [os.path.join(PATH, 'linen/103/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
p_10 = [os.path.join(PATH, 'silk/111/tactile/spectrogram/', "%d.npy" % i) for i in range(480, 540)]
Bfilepaths = p_1 + p_2 + p_3 + p_4 + p_5 + p_6 + p_7 + p_8 + p_9 + p_10


m1 = [os.path.join(PATH, 'velvet/5/visual/', "%d.jpg" % j) for j in range(560, 570)]
m2 = [os.path.join(PATH, 'cotton/27/visual/', "%d.jpg" % j) for j in range(560, 570)]
m3 = [os.path.join(PATH, 'leather/32/visual/', "%d.jpg" % j) for j in range(560, 570)]
m4 = [os.path.join(PATH, 'fiber/45/visual/', "%d.jpg" % j) for j in range(560, 570)]
m5 = [os.path.join(PATH, 'chiffon/57/visual/', "%d.jpg" % j) for j in range(560, 570)]
Cfilepaths = m1 + m2 + m3 + m4 + m5

m_6 = [os.path.join(PATH, 'velvet/4/tactile/spectrogram/', "%d.npy" % j) for j in range(560, 570)]
m_7 = [os.path.join(PATH, 'cotton/27/tactile/spectrogram/', "%d.npy" % j) for j in range(560, 570)]
m_8 = [os.path.join(PATH, 'leather/32/tactile/spectrogram/', "%d.npy" % j) for j in range(560, 570)]
m_9 = [os.path.join(PATH, 'fiber/45/tactile/spectrogram/', "%d.npy" % j) for j in range(560, 570)]
m_10 = [os.path.join(PATH, 'chiffon/57/tactile/spectrogram/', "%d.npy" % j) for j in range(560, 570)]
Dfilepaths = m_6 + m_7 + m_8 + m_9 + m_10



def my_func(txt):
    a = np.load(txt.decode())
    return a.astype(np.float32)

def image_preprocessing(image):

    b = skimage.filters.prewitt_v(image)

    return b.astype(np.float32)

def load(image_name, txt_name):
    # load image data
    image = tf.io.read_file(image_name)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.reshape(image, [1024, 1024])
    txt = tf.compat.v1.py_func(my_func, [txt_name], tf.float32)
    txt = tf.reshape(txt, [257, 11, 1])
    real_data = tf.cast(txt, tf.float32)
    input_image = tf.cast(image, tf.float32)
    input_image = (input_image / 127.5) - 1

    return input_image, real_data

BUFFER_SIZE = 400
BATCH_SIZE = 8

train_image = tf.constant(filepaths)
train_spec = tf.constant(filepaths_1)
train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_spec))
train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


image_name_1 = tf.constant(Afilepaths)
txt_name_1 = tf.constant(Bfilepaths)
test_dataset = tf.data.Dataset.from_tensor_slices((image_name_1, txt_name_1))
test_dataset = test_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
show_dataset = test_dataset.shuffle(400).batch(1)
test_dataset = test_dataset.batch(1)


valid_image = tf.constant(Cfilepaths)
valid_txt = tf.constant(Dfilepaths)
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image, valid_txt))
valid_dataset = valid_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
showtrain_dataset = valid_dataset.shuffle(100).batch(1)
valid_dataset = valid_dataset.batch(1)

sample_trainimage, sample_trainspec = next(iter(showtrain_dataset))
sample_image, sample_spec = next(iter(show_dataset))


#---------------------------------------------------------------------------------------
# build generator
def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return tf.keras.layers.LayerNormalization

def Generator():

    concat_axis = 3
    norm = 'batch_norm'
    Norm = _get_norm_layer(norm)
    initializer = tf.keras.initializers.glorot_normal
    noise = keras.Input(shape=(50, ))

    inputs = keras.Input(shape=(1024, 1024, 1))
    conv0_1 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(inputs)
    conv0_3 = tf.nn.relu(conv0_1)

    conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv0_3)
    conv1_2 = Norm()(conv1_1)
    conv1_3 = tf.nn.relu(conv1_2)

    conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv1_3)
    conv2_2 = Norm()(conv2_1)
    conv2_3 = tf.nn.relu(conv2_2)

    conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv2_3)
    conv3_2 = Norm()(conv3_1)
    conv3_3 = tf.nn.relu(conv3_2)

    conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=4, padding='same', kernel_initializer=initializer, use_bias=False)(conv3_3)
    conv4_2 = Norm()(conv4_1)
    conv4_3 = tf.nn.relu(conv4_2)

    conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=4, padding='same', kernel_initializer=initializer, use_bias=False)(conv4_3)
    conv5_2 = Norm()(conv5_1)
    conv5_3 = tf.nn.relu(conv5_2)

    conv5_4 = tf.keras.layers.Conv2D(512, (3, 3), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv5_3)
    conv5_5 = Norm()(conv5_4)
    conv5_6 = tf.nn.relu(conv5_5)

    h= tf.keras.layers.Conv2D(512, (3, 3), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv5_6)
    h = Norm()(h)
    h = tf.nn.relu(h)

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(units=128)(h)
    latent_code = h

    h = tf.keras.layers.concatenate([h, noise], axis=1, name='concatenation')
    h = tf.keras.layers.Dense(units=1 * 1 * 512)(h)
    h = tf.keras.layers.Reshape((1, 1, 512))(h)

    h = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(h)
    h = Norm()(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    latent = tf.nn.relu(h)


    up = tf.keras.layers.concatenate([conv5_6, latent], axis=concat_axis, name='skip_connection0')
    conv_6_1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False)(up)
    conv_6_2 = Norm()(conv_6_1)
    drop = tf.keras.layers.Dropout(0.5)(conv_6_2)
    conv_6_3 = tf.nn.relu(drop)

    crop_6 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), name='cropped_conv5_2')(conv5_3)
    up6 = tf.keras.layers.concatenate([conv_6_3, crop_6], axis=concat_axis, name='skip_connection1')
    conv6_1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False)(up6)
    conv6_2 = Norm()(conv6_1)
    conv6_2_2 = tf.keras.layers.Dropout(0.5)(conv6_2)
    conv6_3 = tf.nn.relu(conv6_2_2)

    crop_7 = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)), name='cropped_conv4_2')(conv4_1)
    up7 = tf.keras.layers.concatenate([conv6_3, crop_7], axis=concat_axis, name='skip_connection2')
    conv7_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 1), padding='same', kernel_initializer=initializer, use_bias=False)(up7)
    conv7_2 = Norm()(conv7_1)
    conv7_3 = tf.nn.relu(conv7_2)

    h = tf.pad(conv7_3, [[0, 0], [3, 3], [3, 4], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(256, 7, kernel_initializer=initializer, padding='valid')(h)
    conv7_3 = tf.nn.relu(h)

    crop_8 = tf.keras.layers.Cropping2D(cropping=((24, 24), (27, 28)), name='cropped_conv3_2')(conv3_1)
    up8 = tf.keras.layers.concatenate([conv7_3, crop_8], axis=concat_axis, name='skip_connection3')
    conv8_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 1), padding='same', kernel_initializer=initializer, use_bias=False)(up8)
    conv8_2 = Norm()(conv8_1)
    conv8_3= tf.nn.relu(conv8_2)

    h = tf.pad(conv8_3, [[0, 0], [3, 3], [3, 4], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(128, 7, kernel_initializer=initializer, padding='valid')(h)
    conv8_3 = tf.nn.relu(h)

    crop_9 = tf.keras.layers.Cropping2D(cropping=((48, 48), (59, 59)), name='cropped_conv2_2')(conv2_1)
    up9 = tf.keras.layers.concatenate([conv8_3, crop_9], axis=concat_axis, name='skip_connection4')
    conv9_1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 1), padding='same', kernel_initializer=initializer, use_bias=False)(up9)
    conv9_2 = Norm()(conv9_1)
    conv9_3 = tf.nn.relu(conv9_2)

    h = tf.pad(conv9_3, [[0, 0], [3, 3], [3, 4], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(64, 7, kernel_initializer=initializer, padding='valid')(h)
    conv9_3 = tf.nn.relu(h)

    crop_10 = tf.keras.layers.Cropping2D(cropping=((96, 96), (122, 123)), name='cropped_conv1_2')(conv1_1)
    up10 = tf.keras.layers.concatenate([conv9_3, crop_10], axis=concat_axis, name='skip_connection5')
    conv10_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 1), padding='same', kernel_initializer=initializer, use_bias=False)(up10)
    conv10_2 = Norm()(conv10_1)
    conv10_3 = tf.nn.relu(conv10_2)

    crop_11 = tf.keras.layers.Cropping2D(cropping=((192, 192), (250, 251)), name='cropped_conv0_2')(conv0_1)
    up11 = tf.keras.layers.concatenate([conv10_3, crop_11], axis=concat_axis, name='skip_connection6')
    conv11_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 1), padding='same', kernel_initializer=initializer, use_bias=False)(up11)
    conv11_3 = Norm()(conv11_1)
    conv11_4 = tf.nn.relu(conv11_3)

    h = tf.pad(conv11_4, [[0, 0], [3, 4], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(1, 7, kernel_initializer=initializer, padding='valid')(h)
    h = tf.nn.relu(h)


    model = tf.keras.Model(inputs=[inputs, noise], outputs=[h, latent_code])
    model.summary()

    return model

generator = Generator()


LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, generated_image, target):

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - generated_image))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


#---------------------------------------------------------------------------------------
# build discriminator
def Discriminator():
    norm = 'batch_norm'
    Norm = _get_norm_layer(norm)
    initializer = tf.keras.initializers.glorot_normal

    inp = keras.Input((1024, 1024, 1))
    inp_1 = tf.keras.layers.Cropping2D(cropping=((383, 384), (506, 507)), name='cropped_input')(inp)
    tar = keras.Input((257, 11, 1))
    inputs = tf.keras.layers.concatenate([inp_1, tar], axis=3, name='concatenation')

    h = keras.layers.Conv2D(64, 4, strides=(2, 2), kernel_initializer=initializer, padding='same')(inputs)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = keras.layers.Conv2D(128, 4, strides=(2, 2), kernel_initializer=initializer, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = keras.layers.Conv2D(256, 4, strides=(2, 1), kernel_initializer=initializer, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)


    zero_pad1 = tf.keras.layers.ZeroPadding2D()(h)
    h = tf.keras.layers.Conv2D(512, (5, 4), strides=1,
                               kernel_initializer=initializer,
                               use_bias=False)(zero_pad1)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)


    zero_pad2 = tf.keras.layers.ZeroPadding2D()(h)
    h = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)


    model = tf.keras.Model(inputs = [inp, tar], outputs=[h])

    model.summary()

    return model

discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output) # real data tensor -> 1

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output) # fake data tensor -> 0

  total_disc_loss = 0.5*real_loss + generated_loss

  return total_disc_loss


#---------------------------------------------------------------------------------------
critic = 5
clip_value = 0.01
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
checkpoint_dir = './FrictGAN_data/saved_model/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

def generate_output(model, test_input):

  noise = tf.random.normal([1, 50])
  prediction, latent_code = model([test_input, noise], training=True)

  return prediction


def data_save(model, test_input, tar, i):

    noise = tf.random.normal([1, 50])
    prediction, latent_code = model([test_input, noise], training=True)
    pd = tf.reshape(prediction, [257, 11])
    path = './FrictGAN_data/generated_data'
    np.save(path + "/{}.npy".format(i), pd)


def generate_images(model, test_input, tar, i):

  noise = tf.random.normal([1, 50])
  prediction, latent_code = model([test_input, noise], training=True)
  pd = tf.reshape(prediction, [257, 11])
  tar = tf.reshape(tar, [257, 11])
  S = pd.numpy()
  T = tar.numpy()

  y_inv = np.abs(librosa.griffinlim(S, n_iter=32, hop_length=128, win_length=512, window='hamming', center=True, length=None, pad_mode='reflect', momentum=0.99, init=None, random_state=None))
  y_inv1 = np.abs(librosa.griffinlim(T, n_iter=32, hop_length=128, win_length=512, window='hamming', center=True, length=None, pad_mode='reflect', momentum=0.99, init=None, random_state=None))

  plt.figure()
  plt.subplot(3, 1, 1)
  plt.title('input image')
  image = np.reshape(test_input[0], [1024, 1024])
  plt.imshow(image, cmap='gray')

  plt.subplot(3, 1, 2)
  librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))
  plt.title('Generated Signals')
  plt.tight_layout()

  plt.subplot(3, 1, 3)
  librosa.display.specshow(librosa.amplitude_to_db(T, ref=np.max))
  plt.title('Original Signals')
  plt.tight_layout()
  plt.show()

  n = 1280
  x = range(0, n)
  plt.figure(figsize=(5, 2), dpi=120)
  plt.subplot()
  bx = plt.gca()
  bx.plot(x[:1280], y_inv1[:1280], label="Real")
  bx.plot(x[:1280], y_inv[:1280], label="Generated")
  bx.legend()
  bx.yaxis.grid()
  plt.show()



noise_dim = 50
@tf.function
def train_step_gen(input_image, target, epoch):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])
  with tf.GradientTape() as gen_tape:

      gen_output, latent_code = generator([input_image, noise], training=True)

      disc_generated_output = discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)

  generator_gradients = gen_tape.gradient(target=gen_total_loss, sources=generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)


  return gen_total_loss, gen_gan_loss, gen_l1_loss


# discriminator train
def train_step_dis(input_image, target, epoch):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])
  with tf.GradientTape() as disc_tape:

      gen_output, latent_code = generator([input_image, noise], training=True)

      disc_real_output = discriminator([input_image, target], training=True)

      disc_generated_output = discriminator([input_image, gen_output], training=True)

      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  discriminator_gradients = disc_tape.gradient(target=disc_loss, sources=discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  for w in discriminator.trainable_variables:
      w.assign(tf.clip_by_value(w, -clip_value, clip_value))

  with summary_writer.as_default():
      tf.summary.scalar('disc_loss', disc_loss, step=epoch)

  return disc_loss

# training process
def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()

      # WGAN - critic
      for i in range(critic):
         D = train_step_dis(input_image, target, epoch)
      TL, G, L1 = train_step_gen(input_image, target, epoch)

      print("D_loss: {:.2f}".format(D), "G_loss {:.2f}".format(G), "gen_l1_loss {:.2f}".format(L1), "gen_total_loss: {:.2f}".format(TL))

    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, dest='epoch', default=200)
    parser.add_argument('--train', help='train FrictganNet', action='store_true')
    parser.add_argument('--test', help='test FrictganNet', action='store_true')
    parser.add_argument('--visualize', help='visualize FrictganNet outputs', action='store_true')

    args = parser.parse_args()

    if (args.train):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        fit(train_dataset, args.epoch, test_dataset)

    if (args.test):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        i = 0
        for inp, tar in test_dataset.take(600):
           data_save(generator, inp, tar, i)
           i = i+1

    if (args.visualize):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        i = 0
        for inp, tar in show_dataset.take(600):
            generate_images(generator, inp, tar, i)
            i = i+1


if __name__=='__main__':
    main()




