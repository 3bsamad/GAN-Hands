# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:49:34 2020

@author: ctj-oe
"""

import PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf

from platform import python_version_tuple

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
else:
    from itertools import izip, imap

import numpy as np



img_width, img_height = 64, 64

train_data_dir ='F://Omar//tests//faces//data//'
validation_data_dir = 'F://Omar//tests//faces//data//validation//'
nb_train_samples = 56001
nb_validation_samples = 13999
epochs = 50
batch_size = 16


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(
    directory = train_data_dir,
    class_mode = 'categorical',
    batch_size = batch_size)


test_generator = test_datagen.flow_from_directory(
    directory = validation_data_dir,
    class_mode = 'categorical',
    batch_size = batch_size)


train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)








#methods for iterating over the data
for x_train,y_train in train_generator:
    break

model=Sequential()
model.compile(optimizer='Adam', loss=None)
model.fit(
        train_generator,
        validation_data=test_generator)



(train_images, train_labels) = x_train,y_train 
'''
x, y = izip(*(train_generator[i] for i in xrange(len(train_generator))))
x_train, y_train = np.vstack(x), np.vstack(y)

'''
