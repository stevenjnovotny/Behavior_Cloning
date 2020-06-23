# Import Driving Data

import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # columns in line:
        # center camera, right camera, left camera, steering angle, throttle, brake, speed
        samples.append(line)

      
# split data into training and validation sets

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Training samples: %d' % len(train_samples))
print('Validation samples: %d' % len(validation_samples))

# prepare for handling the data
# - include a flippeed version of each image in the test set   

import cv2
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                image_flipped = np.fliplr(center_image)
                angle_flipped = -center_angle
                images.append(image_flipped)
                angles.append(angle_flipped)                
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 32


# compile and train model using generators

from keras.models import Sequential, Model
from keras.layers import Lambda, Dense, Cropping2D, Flatten, Conv2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)
ch, row, col = 3, 160, 320   # image size

model = Sequential()

# preprocessing: normalize, center around zero, stdev = 1.0
# architecture: then use a slightly modified version of the NVIDIA autonomous
# car neural network. 

model.add(Lambda(lambda x: x/127.5 - 1., input_shape = (row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

"""
# using inception
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
input = Input(shape=(row, col, ch)
inception = InceptionV3(weights='imagenet', imclude_top=False, input_shape=(row,col,ch)
for idx, layer in enumeerate(inception.layers):
    layer.trainable = False
input = Input(shape=(    

"""

# add keras checkpoints to reecord model and prevent overfitting

checkpoint = ModelCheckpoint(filepath = './model.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5, mode='min')

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, \
                    steps_per_epoch = np.ceil(len(train_samples)/batch_size), \
                    validation_data = validation_generator, \
                    validation_steps = np.ceil(len(validation_samples)/batch_size), \
                    epochs = 30, verbose = 1, callbacks=[checkpoint, stopper])

print(history_object.history.keys())

# plot the training curve

fig = plt.figure(figsize=(8, 3), dpi=80)
plt.plot(history_object.history['loss'][1:])
plt.plot(history_object.history['val_loss'][1:])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('training.png', bbox_inches='tight')
plt.close(fig)

