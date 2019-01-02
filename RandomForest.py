#!/usr/bin/env python
# coding: utf-8

# # Importing all the necessary modules

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import gzip
from sklearn.svm import SVC
from PIL import Image
import os
from sklearn.datasets import fetch_mldata
from numpy import arange
from sklearn.metrics import confusion_matrix


# # Load MNIST data

# In[2]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# # Confusion Matrix Customization

# In[3]:


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
    


# # Validating and Testing on MNIST dataset

# In[4]:



classifier2 = RandomForestClassifier(n_estimators=200, max_features=50)
model_fit = classifier2.fit(training_data[0], training_data[1])
print(' Validation Accuracy')
print(classifier2.score(validation_data[0], validation_data[1]))
print(' Testing Accuracy')
print(classifier2.score(test_data[0], test_data[1]))


# #  Confusion matrix on MNIST Dataset

# In[5]:



RandomForest_prediction_mnist=model_fit.predict(test_data[0])
conf_mat = confusion_matrix(test_data[1], RandomForest_prediction_mnist)
plot_confusion_matrix(cm           = conf_mat, 
                      normalize    = False,
                      title        = "Confusion Matrix")


# # Testing on USPS Dataset

# In[6]:


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

rf=reformat(USPSTar)


print('USPS dataset Testing accuracy')
y=classifier2.score(USPSMat, USPSTar)
print(y)


# # Confusion matrix on USPS dataset

# In[7]:


RandomForest_prediction_usps=model_fit.predict(USPSMat)
conf_mat1 = confusion_matrix(USPSTar, RandomForest_prediction_usps)
plot_confusion_matrix(cm           = conf_mat1, 
                      normalize    = False,
                      title        = "Confusion Matrix")


# In[8]:


from sklearn.metrics import classification_report
print(classification_report(RandomForest_prediction_usps,
        USPSTar))


# In[ ]:




