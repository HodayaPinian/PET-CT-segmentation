import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, load_model
from keras.layers import Input ,BatchNormalization , Activation 
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers 
from sklearn.model_selection import train_test_split
import os
import nibabel as nib
import cv2 as cv
from keras import backend as K
import glob
#import skimage.io as io
#import skimage.color as color
import random as r
import math
#from nilearn import plotting
from PIL import Image
import re
import nrrd

import sitk
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sd = StandardScaler()
norm = MinMaxScaler()

#U NET

def Convolution(input_tensor,filters):
    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1,1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    return x

def add_map():
    
    return

def model(input_shape, attention_map_bool = False):
    
    inputs = Input((input_shape))#256
    
    if attention_map_bool:
        attention_map =  Input((input_shape))
    
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
        re_attention_map = tf.image.resize(attention_map, conv_4.shape[1:3])
        conv_6 = concatenate([conv_6, conv_4*re_attention_map])
    else:
        conv_6 = concatenate([conv_6, conv_4])
        
    
    upsample_7 = UpSampling2D((2, 2)) (conv_6) #64
    conv_7 = Convolution(upsample_7,256) #64

    if attention_map_bool:
        re_attention_map = tf.image.resize(attention_map, conv_3.shape[1:3])
        conv_7 = concatenate([conv_7, conv_3*re_attention_map])
    else:
        conv_7 = concatenate([conv_7, conv_3])
          
    
    upsample_8 = UpSampling2D((2, 2)) (conv_7) #128    
    conv_8 = Convolution(upsample_8,128) #128  
    
    if attention_map_bool:
        re_attention_map = tf.image.resize(attention_map, conv_2.shape[1:3])
        conv_8 = concatenate([conv_8, conv_2*re_attention_map])
    else:
        conv_8 = concatenate([conv_8, conv_2])
       
    
    upsample_9 = UpSampling2D((2, 2)) (conv_8) #256
    conv_9 = Convolution(upsample_9,64) #256
   
    if attention_map_bool:
        re_attention_map = tf.image.resize(attention_map, conv_1.shape[1:3])
        conv_9 = concatenate([conv_9, conv_1*re_attention_map])
    else:
        conv_9 = concatenate([conv_9, conv_1])
        
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_9) # 256*256*1
    
    
    if attention_map_bool:
        model = Model(inputs=[inputs,attention_map], outputs=[outputs]) 
    else:
        model = Model(inputs=[inputs], outputs=[outputs]) 
    
    return model


InputShape = 256


#build model

Adam = optimizers.Adam(learning_rate=0.0001)
model_PET =  model(input_shape = (InputShape,InputShape,1))
model_PET.compile(optimizer = Adam, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

model_CT = model(input_shape = (InputShape,InputShape,1), attention_map_bool = True)
model_CT.compile(optimizer = Adam, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
model_CT.summary()




#upload data & preprocessing

#### in progress - the basic : 

path = r'nrrd_files\ac'
all_im = glob.glob(r'nrrd_files\ac\*OrgVal.*')
all_mask = glob.glob(r'nrrd_files\ac\*mask.*')
listMask = []
listImage = []
for i in range(len(all_im)):
    
    im, _ = nrrd.read(all_im[i],index_order='C')    
    mask, _ = nrrd.read(all_mask[i],index_order='C')
    #standart scale
    im = (im - np.mean(im)) / np.std(im)
    
   



X_train , X_test, Y_train, Y_test = train_test_split(listImage, listMask, test_size=0.15)


#numpy in list to tf
train_dataset_pet = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) 
test_dataset_pet = tf.data.Dataset.from_tensor_slices((X_test, Y_test)) 

#data augmentation


#fit the model    
PET_model = model_PET.fit(train_dataset_pet, test_dataset_pet, batch_size= 4, epochs = 100, validation_split = 0.15)
spattial_attatopn_map_train = model_PET.evaluate(X_train)
spattial_attatopn_map_test  = model_PET.evaluate(X_test)
CT_model = model_CT.fit([X_train, spattial_attatopn_map_train], Y_train, batch_size = 4, epochs= 100, validation_split=0.15)
results = model_CT.evaluate([X_test, spattial_attatopn_map_test], Y_train,  verbose=1, sample_weight=None, steps=None)































