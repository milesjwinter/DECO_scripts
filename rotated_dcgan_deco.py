from __future__ import division, print_function, absolute_import
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_augmentation import ImageAugmentation
import h5py

# Data loading and preprocessing
f = h5py.File('DCGAN_DECO_Image_Database_64.h5','r')
X = f['train/train_images']
X = np.array(X,dtype='float32')/255.
X = np.reshape(X, newshape=[-1, 64, 64, 1])


z_dim = 100 # Noise data points
total_samples = len(X)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, n_units=8 * 8 * 256)
        x = tflearn.batch_normalization(x)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 8, 8, 256])
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 256, 5, activation='tanh')
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 128, 5, activation='tanh')
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 1, 5, activation='sigmoid')
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tflearn.conv_2d(x, 128, 5, activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.conv_2d(x, 256, 5, activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.conv_2d(x,256, 5, activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.fully_connected(x, 1024, activation='tanh')
        x = tflearn.fully_connected(x, 2)
        x = tf.nn.softmax(x)
        return x

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_90degrees_rotation(rotations=[0, 1, 2, 3])
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_blur(sigma_max=2.0)

# Input Data
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_gen_noise')
input_disc_noise = tflearn.input_data(shape=[None, z_dim], name='input_disc_noise')
input_disc_real = tflearn.input_data(shape=[None, 64, 64, 1], 
                                     data_augmentation=img_aug, name='input_disc_real')

# Build Discriminator
disc_fake = discriminator(generator(input_disc_noise))
disc_real = discriminator(input_disc_real, reuse=True)
disc_net = tf.concat([disc_fake, disc_real], axis=0)
# Build Stacked Generator/Discriminator
gen_net = generator(gen_input, reuse=True)
stacked_gan_net = discriminator(gen_net, reuse=True)

# Build Training Ops for both Generator and Discriminator.
disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_target = tflearn.multi_target_data(['target_disc_fake', 'target_disc_real'],
                                        shape=[None, 2])
disc_model = tflearn.regression(disc_net, optimizer='adam',
                                placeholder=disc_target,
                                loss='categorical_crossentropy',
                                trainable_vars=disc_vars,
                                batch_size=154, name='target_disc',
                                op_name='DISC')

gen_vars = tflearn.get_layer_variables_by_scope('Generator')
gan_model = tflearn.regression(stacked_gan_net, optimizer='adam',
                               loss='categorical_crossentropy',
                               trainable_vars=gen_vars,
                               batch_size=154, name='target_gen',
                               op_name='GEN')

# Define GAN model, that output the generated images.
gan = tflearn.DNN(gan_model)

# Prepare input data to feed to the discriminator
disc_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# Feed to the discriminator (0: fake image, 1: real image)
y_disc_fake = np.zeros(shape=[total_samples])
y_disc_real = np.ones(shape=[total_samples])
y_disc_fake = tflearn.data_utils.to_categorical(y_disc_fake, 2)
y_disc_real = tflearn.data_utils.to_categorical(y_disc_real, 2)

# Prepare input data to feed to the stacked generator/discriminator
gen_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
y_gen = np.ones(shape=[total_samples])
y_gen = tflearn.data_utils.to_categorical(y_gen, 2)


# Start training, feed both noise and real images.
gan.fit(X_inputs={'input_gen_noise': gen_noise,
                  'input_disc_noise': disc_noise,
                  'input_disc_real': X},
        Y_targets={'target_gen': y_gen,
                   'target_disc_fake': y_disc_fake,
                   'target_disc_real': y_disc_real},
        n_epoch=100,
        shuffle=True)

gan.save("deco_gan.tfl")

gen = tflearn.DNN(gen_net, session=gan.session)


f, a = plt.subplots(4, 10, figsize=(10, 4))
for i in xrange(10):
    z = np.random.uniform(-1., 1., size=[4, z_dim])
    g = np.array(gen.predict({'input_gen_noise': z}))
    for j in xrange(4):
        a[j][i].imshow(g[j,:,:,0],cmap=mpl.cm.hot)
        a[j][i].axis('off')

plt.savefig('sample_images_real.png')
