#!/usr/bin/env python
# coding: utf-8

# # Importing the modules

# In[1]:


from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np
import csv
import math
import matplotlib.pyplot
from itertools import islice
from matplotlib import pyplot as plt
import random
import pandas as pd
from keras.utils import np_utils
import pickle
import gzip


# # Extracting features and labels from the dataset

# In[2]:




filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

x = np_utils.to_categorical(np.array(training_data[1]),10)
valTarget = np_utils.to_categorical(np.array(validation_data[1]),10)
valFeature = validation_data[0]
testFeature = test_data[0]
testTarget = np_utils.to_categorical(np.array(test_data[1]),10)


# # Confusion Matrix Customization 

# In[3]:


# import numpy as np


def plot_confusion_matrix(cm,                        
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
   
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

   

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
    


# # Keras Model

# In[4]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard


import numpy as np

input_size = 784
drop_out = 0.2   # to remove overfitting we use dropout
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 10


def get_model():
    
    
    model = Sequential()
    
       
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))     
    model.add(Dropout(drop_out)) 
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax')) 
    
    model.summary()
    
    
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model=get_model()


# # Running the model on training data

# In[5]:


#hyper parameters
# validation_data = 0.1
num_epochs = 300
model_batch_size = 2048
tb_batch_size = 64
early_patience =100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

history = model.fit(training_data[0]
                    , x
                    , validation_data=(valFeature,valTarget)
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# In[6]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # Testing on MNIST validation and testing datasets

# In[7]:


from sklearn.metrics import confusion_matrix
test_loss, test_acc = model.evaluate(testFeature, testTarget)
print('Test Accuracy for test data: ' +str(test_acc))
print('Test Loss for test data: ' +str(test_loss))


val_loss, val_acc = model.evaluate(valFeature, valTarget)
print('Test Accuracy for validation data: ' +str(val_acc))
print('Test Loss for validation target: ' +str(val_loss))


Neural_prediction_mnist = []
for i,j in zip(testFeature,testTarget):
    y = model.predict(np.array(i).reshape(-1,784))
    Neural_prediction_mnist.append((y.argmax()))
    


# # Confusion Matrix of MNIST dataset

# In[8]:


conf_mat = confusion_matrix(test_data[1],Neural_prediction_mnist )
plot_confusion_matrix(cm           = conf_mat, 
                      normalize    = False,
                      title        = "Confusion Matrix")


# # Testing on USPS dataset
# 

# In[9]:


from PIL import Image
import os
import numpy as np

image_size = 28
num_labels = 10


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


def reformat(labels):
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

USPSMat = np.array(USPSMat)
USPSTar= np.array(USPSTar)
test = np_utils.to_categorical(np.array(USPSTar),10)



# # Testing and  Confusion Matrix of USPS dataset

# In[20]:


test_loss1, test_acc1 = model.evaluate(USPSMat, test)
print('Test Accuracy for USPS dataset: ' +str(test_acc1))
print('Test Loss for USPS dataset: ' +str(test_loss1))

Neural_prediction_usps = []
for i,j in zip(USPSMat,USPSTar):
    y = model.predict(np.array(i).reshape(-1,784))
    Neural_prediction_usps.append((y.argmax()))
conf_mat1 = confusion_matrix(USPSTar,Neural_prediction_usps )
plot_confusion_matrix(cm           = conf_mat1, 
                      normalize    = False,
                      title        = "Confusion Matrix")



# In[21]:


from sklearn.metrics import classification_report
print(classification_report(Neural_prediction_usps,
        USPSTar))


# In[ ]:




