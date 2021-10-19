import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import (KNeighborsClassifier, NeighborhoodComponentsAnalysis)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.pipeline import Pipeline



def evaluate_classifier(classifier, test_x, test_y):
    pred_y = classifier.predict(test_x)
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    print(confusion_matrix)
    f1_score = metrics.f1_score(test_y, pred_y, average='macro')
    acc_score = metrics.accuracy_score(test_y, pred_y)
    print('F1: ', f1_score)
    print('Accuracy: ', acc_score)

def k_fold_cross_validation_svm(scaled_train_x, k=5, C=1, kernel='linear', degree=3, gamma='auto'):
    avg_score = 0
    cv = model_selection.KFold(n_splits=k, random_state=0)
    classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    for train_index, test_index in cv.split(scaled_train_x):
        fold_train_x, fold_test_x = x[train_index], x[test_index]
        fold_train_y, fold_test_y = y[train_index], y[test_index]
        classifier.fit(fold_train_x, fold_train_y)
        fold_pred_y = classifier.predict(fold_test_x)
        score = metrics.accuracy_score(fold_test_y, fold_pred_y)
        print(score)
        avg_score += score
    avg_score = avg_score / k
    return avg_score


#### Classificatori, per ognuno sono specificati gli iperparemetri trovati ######

def KNN(train_x_stand, test_x_stand, train_y, test_y):
  k = 5 
  nca = NeighborhoodComponentsAnalysis(random_state=42)
  knn = KNeighborsClassifier(n_neighbors=k)
  nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
  nca_pipe.fit(train_x_stand, train_y)
  y_pred = nca_pipe.predict(test_x_stand)
  test_f1score = f1_score(test_y, y_pred, average='macro')
  accuracy = accuracy_score(test_y, y_pred)
  print ('KNeighborsClassifier f1: ', test_f1score, ' accuracy: ', accuracy, 'k: ',k)

def SVM_rbf(train_x_stand, test_x_stand, train_y, test_y):
  gamma = 0.01
  C = 100
  svc = svm.SVC(kernel='rbf', gamma=gamma, C=C)
  ovr = OneVsRestClassifier(svc)
  ovr.fit(train_x_stand,train_y)
  y_pred = ovr.predict(test_x_stand)
  accuracy = accuracy_score(test_y, y_pred)
  test_f1score = f1_score(test_y, y_pred, average='macro')
  print ('SVM kernel: rbf  startegia: ovr f1 : ', test_f1score, ' accuracy: ', accuracy, ' gamma:' ,gamma, 'C:', C)

def SVM_rb_ovo(train_x_stand, test_x_stand, train_y, test_y):
  gamma = 0.01
  C = 100
  svc = svm.SVC(kernel='rbf', gamma=gamma, C=C)
  svc.fit(train_x_stand,train_y)
  y_pred = svc.predict(test_x_stand)
  accuracy = accuracy_score(test_y, y_pred)
  test_f1score = f1_score(test_y, y_pred, average='macro')
  print ('SVM kernel: rbf  startegia: ovo f1 : ', test_f1score, ' accuracy: ', accuracy, ' gamma:' ,gamma, 'C:', C)

def SVM_poly(train_x_stand, test_x_stand, train_y, test_y):
  degree = 2
  C = 1000
  svc = svm.SVC(kernel='poly', degree=degree, C=C)
  ovr = OneVsRestClassifier(svc)
  ovr.fit(train_x_stand,train_y)
  y_pred = ovr.predict(test_x_stand)
  accuracy = accuracy_score(test_y, y_pred)
  test_f1score = f1_score(test_y, y_pred, average='macro')
  print ('SVM kernel: poly  startegia: ovr f1 : ', test_f1score, ' accuracy: ', accuracy, ' degree:', degree, 'C:', C)
  
def SVM_poly_ovo(train_x_stand, test_x_stand, train_y, test_y):
  degree = 2
  C = 1000
  svc = svm.SVC(kernel='poly', degree=degree, C=C)
  svc.fit(train_x_stand,train_y)
  y_pred = svc.predict(test_x_stand)
  accuracy = accuracy_score(test_y, y_pred)
  test_f1score = f1_score(test_y, y_pred, average='macro')
  print ('SVM kernel: poly  startegia: ovo f1 : ', test_f1score, ' accuracy: ', accuracy, ' degree:', degree, 'C:', C)
  
def SVM_lin(train_x_stand, test_x_stand, train_y, test_y):
  C = 1
  svc = svm.SVC(kernel='linear',  C=C)
  ovr = OneVsRestClassifier(svc)
  ovr.fit(train_x_stand,train_y)
  y_pred = ovr.predict(test_x_stand)
  accuracy = accuracy_score(test_y, y_pred)
  test_f1score = f1_score(test_y, y_pred, average='macro')
  print ('SVM kernel: linear  startegia: ovr f1 : ', test_f1score, ' accuracy: ', accuracy,  'C:', C)
  
def RandomFC(train_x_stand, test_x_stand, train_y, test_y):
  start_time = time.time()
  best_trees = 800
  best_depth = 50
  rfclf = RandomForestClassifier(n_estimators=best_trees, max_depth=best_depth)
  rfclf.fit(train_x_stand,train_y)
  y_pred = rfclf.predict(test_x_stand)
  rf_test_acc = accuracy_score(test_y, y_pred)
  rf_test_f1score = f1_score(test_y, y_pred, average='macro')
  print('RandomForestClassifier f1: ',rf_test_f1score, 'accuracy: ',rf_test_acc, 'n_estimators: ', best_trees, 'depth: ',best_depth)
  
def AdaBoost(train_x_stand, test_x_stand, train_y, test_y):
  best_max_depth = 15
  best_max_features = 15
  best_min_samples_leaf = 9
  best_trees = 1000
  best_lr = 1
  rfclf = AdaBoostClassifier( DecisionTreeClassifier(max_depth=best_max_depth,
  max_features=best_max_features, min_samples_leaf=best_min_samples_leaf), n_estimators=best_trees, learning_rate=best_lr, algorithm="SAMME")
  rfclf.fit(train_x_stand,train_y)
  y_pred = rfclf.predict(test_x_stand)
  rf_test_acc = accuracy_score(test_y, y_pred)
  rf_test_f1score = f1_score(test_y, y_pred, average='macro')
  print('AdaBoostClassifier algorithm: SAMME f1: ',rf_test_f1score, 'accuracy: ',rf_test_acc, 'n_estimators: ',
  best_trees, 'depth: ',best_max_depth, 'max_features: ',  best_max_features, 'min_samples_leaf: ',best_min_samples_leaf, 'lr: ', best_lr  )
  
  
def AdaBoost2(train_x_stand, test_x_stand, train_y, test_y):
  best_max_depth = 15
  best_max_features = 15
  best_min_samples_leaf = 9
  best_trees = 900
  best_lr = 1.5
  rfclf = AdaBoostClassifier( DecisionTreeClassifier(max_depth=best_max_depth,
  max_features=best_max_features, min_samples_leaf=best_min_samples_leaf), n_estimators=best_trees, learning_rate=best_lr, algorithm="SAMME.R")
  rfclf.fit(train_x_stand,train_y)
  y_pred = rfclf.predict(test_x_stand)
  rf_test_acc = accuracy_score(test_y, y_pred)
  rf_test_f1score = f1_score(test_y, y_pred, average='macro')
  print('AdaBoostClassifier algorithm: SAMME.R f1: ',rf_test_f1score, 'accuracy: ',rf_test_acc, 'n_estimators: ',
  best_trees, 'depth: ',best_max_depth, 'max_features: ',  best_max_features, 'min_samples_leaf: ',best_min_samples_leaf, 'lr: ', best_lr )
  
  
def Bagging(train_x_stand, test_x_stand, train_y, test_y):  
  best_depth= 50
  best_trees= 800
  rfclf = BaggingClassifier( DecisionTreeClassifier(max_depth=best_depth), n_estimators=best_trees)
  rfclf.fit(train_x_stand,train_y)
  y_pred = rfclf.predict(test_x_stand)
  rf_test_acc = accuracy_score(test_y, y_pred)
  rf_test_f1score = f1_score(test_y, y_pred, average='macro')
  print('BaggingClassifier f1: ',rf_test_f1score, 'accuracy: ',rf_test_acc, 'n_estimators: ',
  best_trees, 'depth: ',best_depth)
  
def LinearSVC(train_x_stand, test_x_stand, train_y, test_y):
  best_multi_class = 'ovr'
  best_C = 0.1
  svc = svm.LinearSVC( multi_class=best_multi_class, C=best_C)
  svc.fit(train_x_stand,train_y)
  y_pred = svc.predict(test_x_stand)
  svm_test_acc = accuracy_score(test_y, y_pred)
  svm_test_f1score = f1_score(test_y, y_pred, average='macro')
  print('LinearSVC f1: ',svm_test_f1score, 'accuracy: ',svm_test_acc, 'algorithm: ',
  best_multi_class, 'C: ', best_C)