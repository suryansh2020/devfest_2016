#!/usr/bin/env python2
""""
learn.py
Example:
    ./learn.py

Usage:
    learn.py [-f <folders>] [-j <jsons>]
    learn.py -h | --help

Options:
    -h --help           Show this screen.
    -f <folders>		input image folders or json files
    -j <jsons>			input json files

"""

from docopt import docopt
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from svm import SVMClassifier
import json
import os
import numpy as np
import sys


def extractFeatures(folders):
    # pripojeni caffe featur
    return


def vectorize(listOfDicts):
    vec = DictVectorizer()
    ldArray = vec.fit_transform(listOfDicts).toarray()
    ldArray = Imputer().fit_transform(ldArray)
    ldNames = np.asarray(vec.get_feature_names())
    return ldNames, ldArray


def main(argv):
    args = docopt(__doc__)
    folders = args["-f"]
    jsons = args["-j"]

    # byly zadany obrazkove slozky?
    print("Creating dict of features...")
    if folders:
        # extrakce featur a ulozeni do jsonu
        features, labels = extractFeatures(folders)

    # byly zadany json soubory?
    elif jsons:
        maxNrOfFeatures = 200
        features = []
        labels = []
        for path in jsons.split(','):
            with open(path) as f:
                images = json.loads(f.read())["images"]
#            images = images[:100]
#            print(type(images[0]["properties"]))
            for i in images:
                feature = {}
                for e, p in enumerate(i["properties"].keys()):
                    if e == maxNrOfFeatures:
                        break
                    feature[p] = i["properties"][p]
                features.append(feature)
            #features += [i["properties"] for i in images]
            labels += [{"class": i["result"]} for i in images]
        pass
    else:
        print("I don't have either image folders or json files")
        return

    # konverze jsonu na npy pole
    print("Conversion to numpy arrays...")
    XNames, XArray = vectorize(features)
    YNames, YArray = vectorize(labels)

    # svm
    print("SVM learning...")
    svmC = SVMClassifier()
    svmC.run(XArray, YArray)

    # neuronky

    # ulozeni vyslednych jsonu

if __name__ == '__main__':
    main(sys.argv[1:])
