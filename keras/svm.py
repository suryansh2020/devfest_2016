#!/usr/bin/env python
""""
svm.py
Example:
    ./svm.py

Usage:
    svm.py -i <files>
    svm.py -h | --help

Options:
    -h --help           Show this screen.
    -i <files>          json files with img properties and result classes

"""

import pandas as pd
import numpy as np
import sklearn
import subprocess
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import svm
from sklearn import neighbors, datasets
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import argparse
from docopt import docopt
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
#  from sklearn.preprocessing import StandardScaler
import sys
import json


class SVMClassifier():

    def __init__(self):  # , labels, features):
        self.classifiers = {
            'kNN': KNeighborsClassifier(3),
            'SVM': SVC(),
            # SVC(kernel="linear", C=0.025),
            # SVC(gamma=2, C=1),
            'Decision tree': DecisionTreeClassifier(max_depth=5),
            'Random forest': RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1),
            'AdaBoost': AdaBoostClassifier(),
            'Gaussian': GaussianNB()
        }

        # X_train, y_train, X_test, y_test = self.split(
        #     self.features, self.labels)
        # for name in names:
        #     print(name)
        #     clf = self.train(X_train, y_train, name)
        #     print("trained..")
        #     self.test(clf, X_test, y_test)

    def run(self, features, labels):
        if len(labels[0]) > 1:
            YArray_new = []
            for label in labels:
                YArray_new.append(np.where(label == 1)[0][0])
            labels = np.array(YArray_new)

        X_train, y_train, X_test, y_test = self.split(features, labels)

        clf = SVC().fit(X_train, y_train)
        self.test(clf, X_test, y_test)

        # for name, clf in self.classifiers.items():
        #    print name
        #    clf = clf.fit(X_train, y_train)
        #    print("trained")
        #    self.test(clf, X_test, y_test)

        # for clf in self.classifiers:
        #    print(clf)
        #    self.train(X_train, y_train, clf)
        #    print("trained..")
        #    self.test(clf, X_test, y_test)
        #    print("tested")

    def vectorize(self, listOfDicts):
        self.vec = DictVectorizer()
        ldArray = self.vec.fit_transform(listOfDicts).toarray()
        ldArray = Imputer().fit_transform(ldArray)
        ldNames = np.asarray(self.vec.get_feature_names())
        return ldNames, ldArray

    def test(self, clf, X_test, y_test):
        prediction = clf.predict(X_test)

        FP = 0
        TP = 0
        for key, res in enumerate(prediction):
            if res != y_test[key] and y_test[key] == 0:
                FP += 1
            elif res == y_test[key] and y_test[key] == 1:
                TP += 1

        print("test samples: " + str(len(y_test)))
        print("FP: " + str(FP))
        print("TP: " + str(TP))
        print("precision: " + str(TP / float(TP + FP)))
        print("recall: " + str(TP / float(list(y_test).count(1))))

        from sklearn.metrics import confusion_matrix
        labels = ['TP', 'FP']

        print(y_test)
        print(prediction)

        cm = confusion_matrix(y_test, prediction)
        print(cm)

        # GET STATISTICS
        report = ""
        report += "-----------------------------------------------------\n"
        report += "------------ CLASSIFICATOR STATISTICS ---------------\n"
        report += "-----------------------------------------------------\n\n"
        report += 'Precision:\t%s\n' % sklearn.metrics.precision_score(
            y_test, prediction)
        report += 'Accuracy:\t%s\n\n' % sklearn.metrics.accuracy_score(
            y_test, prediction)
        report += "-----------------------------------------------------\n"
        report += "------------ STATISTICS PER CATEGORY ----------------\n"
        report += "-----------------------------------------------------\n\n"
        report += classification_report(y_test, prediction)

        print(report)

    def train(self, X_train, y_train, clf):
        clf.fit(X_train, y_train)

    def split(self, data, classes):
        X = np.array(data)
        y = np.array(classes)
        stratifiedSplit = StratifiedShuffleSplit(
            y, test_size=0.1, random_state=0)

        for train_index, test_index in stratifiedSplit:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, y_train, X_test, y_test


def main(argv):
    args = docopt(__doc__)
    imagePaths = args["-i"]

    imgsProperties = []
    imgsResults = []

    for p in imagePaths.split(","):
        with open(p) as f:
            images = json.loads(f.read())["images"]

        print(p + ": images nr = " + str(len(images)))

        imgsProperties += [i["properties"] for i in images]
        imgsResults += [{"class": i["result"]} for i in images]

    svm = SVMClassifier()

    XNames, XArray = svm.vectorize(imgsProperties)
    YNames, YArray = svm.vectorize(imgsResults)

    print(XArray, YNames)

    svm.run(XArray, YArray)

    return

    # parser = argparse.ArgumentParser()
    # parser.add_argument("features")
    # parser.add_argument("labels")
    # args = parser.parse_args()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("folders")

    # svmClass = SVMClassifier(args.features, args.labels)

if __name__ == "__main__":
    main(sys.argv[1:])
