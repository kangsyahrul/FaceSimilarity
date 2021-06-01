#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import PIL


base_dir = 'ktp'

train_dir = os.path.join( base_dir, 'train')
validation_dir = os.path.join( base_dir, 'test')


train_cats_dir = os.path.join(train_dir, 'ktp') 
train_dogs_dir = os.path.join(train_dir, 'non') 
validation_cats_dir = os.path.join(validation_dir, 'ktp') 
validation_dogs_dir = os.path.join(validation_dir, 'non')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

from keras.models import load_model
model = load_model('bangkitrevisi.h5')
 
# predicting images
path = '../ktp/ktp1.jpg'
ktp = cv2.imread(path)
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)

if classes == 0:
    print("KTP")
    cv2.imshow('KTP', ktp)
    cv2.waitKey(0)
else:
    print("Non KTP")
    cv2.imshow('NonKTP', ktp)
    cv2.waitKey(0)

