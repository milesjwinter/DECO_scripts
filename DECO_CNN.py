from __future__ import print_function
import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, MaxoutDense, AveragePooling2D, Lambda, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import time
from keras import backend as K

start_time = time.time()

def smooth_labels(y, smooth_factor):
    '''
    Convert a matrix of one-hot row-vector labels into smoothed versions.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
    return y


#model/training parameters
batch_size = 32
num_classes = 3 
epochs = 6000

# input image parameters
img_rows, img_cols = 64, 64
img_channels = 1 

#load images and labels
f = h5py.File('DECO_Image_Database.h5','r')
images = np.array(f['train/train_images'],dtype='float32')
labels = np.array(f['train/train_labels'])

#normalize images
images = images/255.

#split into testing and training sets
train_images, test_images, train_labels, test_labels = train_test_split(images, 
                                                                        labels,
                                                                        train_size=0.9, 
                                                                        random_state=123)

f.close()

class_weight = {0:1.0,1:7.0,2:1.85}

#define model structure
model = Sequential()
model.add(Cropping2D(cropping=18,input_shape=(100, 100, img_channels)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2048))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(2048))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

# checkpoint
filepath = "best_checkpointed_model.h5"
checkpointer = ModelCheckpoint(filepath, 
                               monitor='val_loss', 
                               verbose=0, 
                               save_best_only=True, 
                               mode='auto')

#preprocess images
datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.08,
                             height_shift_range=0.08,
                             rotation_range=180.,
                             zoom_range=[0.9,1.1],
                             fill_mode="constant",
                             cval=0)

#fit the model
datagen.fit(train_images)
history = model.fit_generator(datagen.flow(train_images, train_labels,
                    batch_size=batch_size),
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs,
                    class_weight=class_weight,
                    validation_data=(test_images, test_labels),
                    callbacks=[checkpointer],
                    initial_epoch=2000)

#save model weights and structure
model.save('trained_model.h5')

#evaluate  model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("job time: ", time.time() - start_time)

# list all data in history
print(history.history.keys())
history_vals = np.array([history.history['acc'],
                         history.history['val_acc'],
                         history.history['loss'],
                         history.history['val_loss']])
np.savetxt('history_vals.txt',np.transpose(history_vals))


