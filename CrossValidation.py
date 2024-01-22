#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-
from sklearn.datasets import load_svmlight_file
import random
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_files
import pylab as pl


def main(data):
    lda = LinearDiscriminantAnalysis()
    perceptron = Perceptron()
    linearSVM = LinearSVC(dual = True, max_iter=5000)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(max_iter= 5000)

    X_data, y_data = load_svmlight_file(data)
    X_data = X_data.toarray()

    rsfk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state = random.randint(0, 368512346))
    for i, (train, test) in enumerate (rsfk.split(X_data, y_data)):
        X_train = X_data[train]
        y_train = y_data[train]
        X_test = X_data[test]
        y_test = y_data[test]

        lda = LinearDiscriminantAnalysis()
        perceptron = Perceptron()
        linearSVM = LinearSVC(dual = True, max_iter=5000)
        knn = KNeighborsClassifier()
        lr = LogisticRegression(max_iter= 5000)

        print ('Fitting...')
        lda.fit(X_train, y_train)
        perceptron.fit(X_train, y_train)
        linearSVM.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        print ('Predicting...')
        lda_pred = lda.predict(X_test)        
        perceptron_pred = perceptron.predict(X_test)
        linearSVM_pred = linearSVM.predict(X_test)
        knn_pred = knn.predict(X_test)
        lr_pred = lr.predict(X_test)

        print ('Accuracy LDA: ',  accuracy_score(y_test, lda_pred))
        print ('Accuracy Perceptron: ',  accuracy_score(y_test, perceptron_pred))
        print ('Accuracy Linear SVM: ',  accuracy_score(y_test, linearSVM_pred))
        print ('Accuracy Knn: ',  accuracy_score(y_test, knn_pred))
        print ('Accuracy LR: ',  accuracy_score(y_test, lr_pred))

        

if __name__ == "__main__":

   ##main("meses/libsvm/data_VGG.txt") ##Se usando as features do VGG
   ##main("meses/libsvm/data_Xception.txt") ##Se usando as features do Xception
    main("meses/libsvm/data_Inception.txt") ##Se usando as features do Inception
   


