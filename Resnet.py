#!/usr/bin/env python
# coding: utf-8

# In[21]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Softmax, BatchNormalization
from tensorflow.keras import datasets


# In[22]:


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
res_blocks = 10


# In[65]:


inputs = keras.Input(shape = (32,32,3))
l = layers.Conv2D(32, (3,3), activation = 'relu' )(inputs)
l = layers.BatchNormalization()(l)
l = layers.Conv2D(32, (3,3), activation = 'relu')(l)
l = layers.BatchNormalization()(l)
l = layers.MaxPooling2D(2,2)(l)
#print('typel = ', type(l))

for i in range(res_blocks):
    #print("i = ", i)
    #print("inside res blocks shape of  l = ", l.shape)
    
    l = res_netblocks(i,l ,32, (3,3))
    
l = layers.Conv2D(64, 3, activation='relu')(l)
l = layers.GlobalAveragePooling2D()(l)
l = layers.Dense(256, activation='relu')(l)
l = layers.Dropout(0.5)(l)
outputs = layers.Dense(10, activation='softmax')(l)
res_net_model = keras.Model(inputs, outputs)
    





    


# In[66]:


def res_netblocks(i, inputsx, no_filters, filter_size):
    #print("i = ", i)
    #print("type = ", type(inputsx))
    l = layers.Conv2D(no_filters, (filter_size), activation = 'relu', padding = 'same' )(inputsx)
    l = layers.BatchNormalization()(l)
    l = layers.Conv2D(no_filters, (filter_size), activation = None , padding  = 'same')(l)
    l = layers.BatchNormalization()(l)
    #print(" shape of inputx = ", inputsx.shape)
    #print("shape of rl = ", l.shape)
    l = layers.Add()([l, inputsx])
    l = layers.Activation('relu')(l)
    #print("shape of l after skip connection = ", l.shape)
    
    return  l
    
    
   
    
    
    


# In[70]:


res_net_model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

res_net_model.fit(x_train,y_train, epochs=30)


# In[ ]:




