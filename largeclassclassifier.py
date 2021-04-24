#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:34:34 2021

@author: eseogunje
"""

#  libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# loading the class dataset (80)
data = pd.read_csv('merged_class.csv')



# X -> features, y -> label
columns = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
       'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
       'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
       'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
       'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
       'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
       'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
       'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
       'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
       'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
       'scissors', 'teddy bear', 'hair drier', 'toothbrush']
      
seed = 0
X = data[columns]
y = data.Category

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_models = DecisionTreeClassifier(max_depth = 5,random_state=seed).fit(X_train, y_train)
dtree_predictions = dtree_models.predict(X_test)
dtree_accuracy = dtree_models.score(X_test, y_test)
# creating a confusion matrix
cm1 = confusion_matrix(y_test, dtree_predictions)
print(cm1,dtree_accuracy)



# training a  SVM classifier
from sklearn.svm import SVC

svm_model = SVC(kernel = 'rbf', C = 10,random_state=seed).fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# model accuracy for X_test
svm_accuracy = svm_model.score(X_test, y_test)

# creating a confusion matrix
cm2 = confusion_matrix(y_test, svm_predictions)
print(cm2,svm_accuracy)

# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)

# accuracy on X_test
knn_accuracy = knn.score(X_test, y_test)

# creating a confusion matrix
knn_predictions = knn.predict(X_test)
cm3 = confusion_matrix(y_test, knn_predictions)
print (cm3,knn_accuracy)


# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)

# accuracy on X_test
gnb_accuracy = gnb.score(X_test, y_test)


# creating a confusion matrix
cm4 = confusion_matrix(y_test, gnb_predictions)
print (cm4,gnb_accuracy)

def classifier(data):
    print("SVM Prediction: {}".format(svm_model_linear.predict(data)))
    print("Decision Tree Prediction: {}".format(dtree_models.predict(data)))
    print("KNN Prediction: {}".format(knn.predict(data)))
    print("Naive Bayes Prediction: {}".format(gnb.predict(data)))
    
    



