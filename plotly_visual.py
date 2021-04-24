#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:13:39 2021

@author: eseogunje
"""

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
from mlxtend.plotting import plot_decision_regions

# loading the class dataset (80)
data2 = pd.read_csv('supermerged_class (1).csv')



# X -> features, y -> label
columnss = ['person', 'vehicle', 'outdoor', 'animal', 'accessory', 'sports',
       'kitchen', 'food', 'furniture', 'electronics', 'appliance', 'indoor']
      

Xs = data2[columnss]
ys = data2.Category

# dividing X, y into train and test data
X_trains, X_tests, y_trains, y_tests = train_test_split(Xs, ys, random_state = seed)

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtrees_models = DecisionTreeClassifier(max_depth = 5,random_state=seed).fit(X_trains, y_trains)
dtrees_predictions = dtrees_models.predict(X_tests)
dtrees_accuracy = dtrees_models.score(X_tests, y_tests)
# creating a confusion matrix
cms1 = confusion_matrix(y_tests, dtrees_predictions)
print(cms1,dtrees_accuracy)



# training a  SVM classifier
from sklearn.svm import SVC

svms_model = SVC(kernel = 'rbf', C = 10,random_state=seed).fit(X_trains, y_trains)
svms_predictions = svms_model.predict(X_tests)

# model accuracy for X_test
svms_accuracy = svms_model.score(X_tests, y_tests)

# creating a confusion matrix
cms2 = confusion_matrix(y_tests, svms_predictions)
print(cms2,svms_accuracy)

# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knns = KNeighborsClassifier(n_neighbors = 5).fit(X_trains, y_trains)

# accuracy on X_test
knns_accuracy = knns.score(X_tests, y_tests)

# creating a confusion matrix
knns_predictions = knns.predict(X_tests)
cms3 = confusion_matrix(y_tests, knns_predictions)
print (cms3,knns_accuracy)


# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnbs = GaussianNB().fit(X_trains, y_trains)
gnbs_predictions = gnbs.predict(X_tests)

# accuracy on X_test
gnbs_accuracy = gnbs.score(X_tests, y_tests)


# creating a confusion matrix
cms4 = confusion_matrix(y_tests, gnbs_predictions)
print (cms4,gnbs_accuracy)

def largeclassifier(data):
    print("SVM Prediction: {}".format(svm_model_linear.predict(data)))
    print("Decision Tree Prediction: {}".format(dtree_models.predict(data)))
    print("KNN Prediction: {}".format(knn.predict(data)))
    print("Naive Bayes Prediction: {}".format(gnb.predict(data)))
    




