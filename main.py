#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import timeit
from sklearn import metrics
import scipy.sparse
import NeuralNetwork
import RandomForest
import SVM
import LogisticRegression



def getAccuracy12(output, actualTarget):
    count =0
    for i in range(0,output.shape[0]):
        if output[i]==actualTarget[i]:
            count+=1
    #print(count)
    accuracy = (count/output.shape[0])*100
    return accuracy 

ensemble_test = np.c_[Neural_prediction_mnist,logistic_prediction_mnist,RandomForest_prediction_mnist,SVM_prediction_mnist]
list=[]
from scipy import stats
for i in range(ensemble_test.shape[0]):
    a,b = stats.mode(ensemble_test[i])
    list.append(a[0])

print('Testing Accuracy: ', getAccuracy12(test_data[1],list))


# In[ ]:


ensemble_test1 = np.c_[Neural_prediction_usps,logistic_prediction_usps,RandomForest_prediction_usps,SVM_prediction_usps]
print(ensemble_test1.shape)

list1=[]
from scipy import stats
for i in range(ensemble_test1.shape[0]):
    a,b = stats.mode(ensemble_test1[i])
    list1.append(a[0])
print('Testing Accuracy: ', getAccuracy12(USPSTar,list1))

