import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import  BatchNormalization , Activation #,Input
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import optimizers, layers
from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import glob
#import skimage.io as io
#import skimage.color as color
import random as r
import math
#from nilearn import plotting
from PIL import Image
import re
import nrrd
from keras import backend as K

import SimpleITK as sitk
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sd = StandardScaler()
norm = MinMaxScaler()


#U NET

def Convolution(input_tensor,filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding = "same", strides=(1,1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    return x

def model(input_shape, attention_map_bool = False):
    inputs = Input((input_shape),name='body')#256*256*1
    
    if attention_map_bool:
        attention_map = Input((input_shape), name = 'pet adittion')
    
    conv_1 = Convolution(inputs,64) #256
    maxp_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_1) #128
    conv_2 = Convolution(maxp_1,128) #128
    maxp_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_2) #64
    conv_3 = Convolution(maxp_2,256) #64
    maxp_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_3) #32
    conv_4 = Convolution(maxp_3,512)#32
    maxp_4 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_4) #16
    conv_5 = Convolution(maxp_4,1024) #16
    upsample_6 = UpSampling2D((2, 2)) (conv_5) #32
    conv_6 = Convolution(upsample_6,512) #32
    
    if attention_map_bool:
        re_attention_map = Conv2D(512,(1,1))(tf.image.resize(attention_map, (32,32)))
        conv_6 = concatenate([conv_6, conv_4*re_attention_map])
    else:
        conv_6 = concatenate([conv_6, conv_4])
    upsample_7 = UpSampling2D((2, 2)) (conv_6) #64
    conv_7 = Convolution(upsample_7,256) #64
    
    if attention_map_bool:
        re_attention_map = Conv2D(256,(1,1))(tf.image.resize(attention_map, (64,64)))
        conv_7 = concatenate([conv_7, conv_3*re_attention_map])
    else:
        conv_7 = concatenate([conv_7, conv_3])
    upsample_8 = UpSampling2D((2, 2)) (conv_7) #128
    conv_8 = Convolution(upsample_8,128) #128
    
    if attention_map_bool:
        re_attention_map = Conv2D(128,(1,1))(tf.image.resize(attention_map, (128,128)))
        conv_8 = concatenate([conv_8, conv_2*re_attention_map])
    else:
        conv_8 = concatenate([conv_8, conv_2])
    upsample_9 = UpSampling2D((2, 2)) (conv_8) #256
    conv_9 = Convolution(upsample_9,64) #256
    
    if attention_map_bool:
        re_attention_map = Conv2D(64,(1,1))(tf.image.resize(attention_map, (256,256)))
        conv_9 = concatenate([conv_9, conv_1*re_attention_map])
    else:
        conv_9 = concatenate([conv_9, conv_1])
    
    outputs = Conv2D(36, (1, 1), activation='sigmoid') (conv_9) # 256*256*1
    
    if attention_map_bool:
        model = Model(inputs=[inputs,attention_map], outputs=[outputs])
    else:
        model = Model(inputs=[inputs], outputs=[outputs])
    return model


InputShape = 256
im_size = 36

def upload_im(path):
    im, _ = nrrd.read(path,index_order='C')
    return np.float64(im)

def cutedge(mask, im):
    b = np.nonzero(mask)
    im_size = 36
    #we want the segment area
    min_edge = min(b[2])
    max_edge = max(b[2])
    
    if (max_edge - min_edge)%2:
        rang = int(np.ceil((im_size - (max_edge - min_edge))/2))
        mask = mask[:, :, min_edge - rang +1 : max_edge + rang]
        im = im[:, :, min_edge - rang +1: max_edge + rang]
    else:
        rang = int(np.ceil((im_size - (max_edge - min_edge))/2))
        mask = mask[:, :, min_edge - rang : max_edge + rang]
        im = im[:, :, min_edge - rang: max_edge + rang]
    return mask, im

#data augmentation
class Augment(tf.keras.layers.Layer):
    
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
    
    def call2(self, inputs):
        inputs = self.augment_inputs(inputs)
        return inputs

    
    
    
BATCH_SIZE = 25
BUFFER_SIZE = 1000
Adam = optimizers.Adam(learning_rate=0.0001)

# %%
#upload data & preprocessing

path = r'nrrd_files\ac'  #ac == pet images
all_im = glob.glob(r'nrrd_files\ac\*image.*')
all_mask = glob.glob(r'nrrd_files\ac\*mask.*')
listMask = []
listImage = []


for i in range(int(len(all_im)/2)):
    im = upload_im(all_im[i])
    mask = upload_im(all_mask[i])
    #standart scale --> z score across 3d image
    im = (im - np.mean(im)) / np.std(im)
    mask, im = cutedge(mask, im)
    listMask.append(cv.resize(mask,(InputShape,InputShape)))
    listImage.append(cv.resize(im,(InputShape,InputShape)))


X_train , X_test, Y_train, Y_test = train_test_split(listImage, listMask, test_size=0.15, random_state = 0)

#numpy list to tf
train_dataset_pet = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) 
train_batches_pet = (
    train_dataset_pet
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment().call)
    .prefetch(buffer_size=tf.data.AUTOTUNE))


test_dataset_pet = tf.data.Dataset.from_tensor_slices((X_test, Y_test)) 
test_batches_pet = test_dataset_pet.batch(BATCH_SIZE)

model_PET = model(input_shape = [InputShape,InputShape,36])
model_PET.compile(optimizer = Adam, 
                  loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
#tf.keras.utils.plot_model(model, show_shapes=True)
PET_model = model_PET.fit(train_batches_pet, batch_size = 4, epochs = 5, validation_data=test_batches_pet)

spattial_attatopn_map_train = model_PET.predict(train_batches_pet)
spattial_attatopn_map_test = model_PET.predict(test_batches_pet)

plt.Figure()
plt.imshow(spattial_attatopn_map_train[1,:,:,20], 'gray')




# %% preprocessing data for ct model

spattial_attatopn_map_train = tf.data.Dataset.from_tensor_slices(spattial_attatopn_map_train)
spattial_attatopn_train_batches_ct = (
    spattial_attatopn_map_train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment().call2)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .as_numpy_iterator())



spattial_attatopn_map_test = tf.data.Dataset.from_tensor_slices(spattial_attatopn_map_test)
spattial_attatopn_test_batches_ct = spattial_attatopn_map_test.batch(BATCH_SIZE).as_numpy_iterator()


label_train_pet = tf.data.Dataset.from_tensor_slices((Y_train)) 
label_train_pet = (
    label_train_pet
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment().call2)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

train_pet_for_ct  =  tf.data.Dataset.from_tensor_slices((X_train)) 
train_pet_for_ct = (
    train_pet_for_ct
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment().call2)
    .prefetch(buffer_size=tf.data.AUTOTUNE))


# %% CT

path = r'nrrd_files\ct'
all_im = glob.glob(r'nrrd_files\ct\*image.*')
all_mask = glob.glob(r'nrrd_files\ct\*mask.*')
listMask_ct = []
listImage_ct = []


for i in range(int(len(all_im)/2)):
    im = upload_im(all_im[i])
    mask = upload_im(all_mask[i])
    #standart scale --> z score across 3d image
    im = (im - np.mean(im)) / np.std(im)
    mask, im = cutedge(mask, im)
    listMask_ct.append(cv.resize(mask, (InputShape, InputShape)))
    listImage_ct.append(cv.resize(im, (InputShape, InputShape)))



X_train_ct , X_test_ct, Y_train_ct, Y_test_ct = train_test_split(listImage_ct, listMask_ct, test_size=0.15, random_state= 0)


train_dataset_ct = tf.data.Dataset.from_tensor_slices((X_train_ct))
train_batches_ct = (
    train_dataset_ct
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment().call2)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .as_numpy_iterator())



label =  tf.data.Dataset.from_tensor_slices((Y_train_ct)) 
label_batch = (
    label
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment().call2)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .as_numpy_iterator()) 

test_dataset_ct = tf.data.Dataset.from_tensor_slices((X_test_ct)) 
test_batches_ct = test_dataset_ct.batch(BATCH_SIZE).as_numpy_iterator()

label_test_ct = tf.data.Dataset.from_tensor_slices((Y_test_ct)) 
label_test_ct = label_test_ct.batch(BATCH_SIZE).as_numpy_iterator()


## in tensorflow model with 2 inputs & one label you need numpy array;

#ct data
train_batches_ct = np.array(list(train_batches_ct)[0])
label_batch = np.array(list(label_batch)[0])

test_batches_ct = np.array(list(test_batches_ct)[0])
label_test_ct = np.array(list(label_test_ct)[0])

#pet data
train_label_pet =  np.array(list(spattial_attatopn_train_batches_ct)[0])

test_batches_pet = np.array(list(spattial_attatopn_test_batches_ct)[0])
label_train_pet = np.array(list(label_train_pet)[0])




#fit the model
model_CT = model(input_shape = [InputShape, InputShape, 36], attention_map_bool = True)
model_CT.compile(optimizer = Adam, 
                  loss= [tf.keras.losses.BinaryCrossentropy()],
                  metrics=['accuracy'])

CT_model = model_CT.fit((train_batches_ct, train_label_pet),(label_batch),
                        batch_size = 4,
                        epochs= 5,)


