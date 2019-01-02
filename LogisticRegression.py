#!/usr/bin/env python
# coding: utf-8

# # Importing all the necessary modules

# In[1]:


import pickle
import gzip
from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np
import csv
import math
import matplotlib.pyplot
from itertools import islice
from matplotlib import pyplot as plt
import random


# ## Load MNIST on Python 3.x

# In[2]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# # Confusion matrix customization

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
    
    
    


# # Logistic Regression with multiple classes 

# In[4]:


import scipy.sparse
import tensorflow as tf
def GetSigmoid(z):    
    return 1 / (1 + np.exp(-z))
def SigmoidValTest(VAL_PHI,W):
    z = np.dot(np.transpose(VAL_PHI),W)
    h = GetSigmoid(z)
    return h

def getLoss(w,x,y):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = GetSigmoid(scores) #Next we perform a softmax on these scores to get their probabilities
    #loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad =  np.dot(x.T,( prob-y_mat))#And compute the gradient for that loss
    return grad

def oneHotIt(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX


def getProbsAndPreds(someX):
    probs = GetSigmoid(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds
def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

from beautifultable import BeautifulTable 
AccuracyTable = BeautifulTable()
AccuracyTable.column_headers = ["Learning Rate","Iteration", "AccuracyTraining","AccuarcyValidation","AccuracyTesting"]


W       = tf.Variable(tf.random_normal([784,10],stddev=0.01))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
w = W.eval(sess)

itr=500
learningRate = 0.0001
val   = []
tr   = []
test  = []

W_Mat        = []
losses = []
x=training_data[0]
y=training_data[1]
testX=test_data[0]
testY=test_data[1]


  
for i in range(0,itr):
    grad = getLoss(w,x,y)
    w = w - (learningRate * grad)
    tr.append(getAccuracy(x,y))
    test.append(getAccuracy(testX,testY))
    val.append(getAccuracy(validation_data[0],validation_data[1]))
AccuracyTable.append_row([learningRate,i,str(float(getAccuracy(x,y))),str(float(getAccuracy(validation_data[0],validation_data[1]))),str(float( getAccuracy(testX,testY)))])
print(AccuracyTable)


# # Confusion Matrix of MNIST dataset

# In[5]:


from sklearn.metrics import confusion_matrix
predicted=np.dot(testX,w)
pred=[]
for i in predicted:
    pred.append(i.argmax())
    
logistic_prediction_mnist=np.array(pred)

conf_mat = confusion_matrix(testY,logistic_prediction_mnist )
plot_confusion_matrix(cm           = conf_mat, 
                       normalize    = False,
                     title        = "Confusion Matrix")


# # Testing on USPS dataset

# In[6]:


from PIL import Image
import os
import numpy as np
from keras.utils import np_utils
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
print(USPSMat.shape)
USPSTar= np.array(USPSTar)
print(reformat(USPSTar).shape)
test = np_utils.to_categorical(np.array(USPSTar),10)

getAccuracy(USPSMat,USPSTar)


# # Confusion matrix of USPS dataset

# In[7]:


from sklearn.metrics import confusion_matrix
predicted=np.dot(USPSMat,w)
pred=[]
for i in predicted:
    pred.append(i.argmax())
    
logistic_prediction_usps=np.array(pred)


conf_mat = confusion_matrix(USPSTar,logistic_prediction_usps)
plot_confusion_matrix(cm           = conf_mat, 
                      normalize    = False,
                      title        = "Confusion Matrix")


# In[8]:


from sklearn.metrics import classification_report
print(classification_report(logistic_prediction_usps,
        USPSTar))


# In[ ]:




