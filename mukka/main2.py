'''
Created on ../../2017
@author: Madhura Dole
'''

import librosa
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import  Perceptron
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def main():
    DIR = "genres"


featuresArray = []


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def classifyPerceptron():
    global clf, scoreMLP
    # MLF2
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08)
    clf.fit(np.array(finalList), np.array(classList))
    scoreMLP = clf.score(testFinalList, testClassList)
    print("Multilayer Perceptron: ", scoreMLP)

    # Multilayer Perceptron Classification
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08)
    clf.fit(np.array(finalList), np.array(classList))
    scoreMLPPredict = clf.predict(testFinalList)
    scoreMLP = clf.score(testFinalList, testClassList)
    # recall_MLP = recall_score(testClassList, scoreMLPPredict)
    precision_MLP = precision_score(testClassList, scoreMLPPredict)
    confusion_matrix_MLP = confusion_matrix(testClassList, scoreMLPPredict)

    print("Multilayer Perceptron score: ", scoreMLP)
    # print("Multilayer Perceptron recall : ", recall_MLP)
    print("Multilayer Perceptron precision: ", precision_MLP)
    print("Multilayer Perceptron CM mat: ", confusion_matrix_MLP)


def classfiySVM():
    svm1 = svm.LinearSVC(C=10, loss='squared_hinge', penalty='l2', tol=0.00001)
    svm1.fit(finalList, classList)
    scoreSVM = svm1.score(testFinalList, testClassList)
    print("Support Vector Machines: ", scoreSVM)


def classifyKMeans():
    kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                    verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    kmeans.fit(finalList, classList)
    scoreKMeans = kmeans.score(testFinalList, testClassList)
    print("KMeans: ", scoreKMeans)

def classifyPerceptronOnly():
	
	clf_p = Perceptron(penalty=None, alpha=0.01, fit_intercept=True, n_iter=5000, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
	
	clf_p.fit(finalList, classList)
	res1 = clf_p.predict(finalList)
	score_p = clf_p.score(testFinalList, testClassList)
	print(res1, "predicted perceptron output")
	print(score_p,"pecerptron score")
	
		
def classifyNB():
	mnb = MultinomialNB(alpha = 1.0, fit_prior=True,class_prior=None)
    # acc = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=500, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
	mnb.fit(finalList, classList)
	r = mnb.predict(testfinalList)
	score_mnb = mnb.score(testFinalList, testClassList)
	print(r, "predicted MNB output")
	print(score_mnb,"MNB score")
	
    
	
	
def classifyLogisticRegression():
    acc = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                             class_weight=None, random_state=None, solver='liblinear', max_iter=500, multi_class='ovr',
                             verbose=0, warm_start=False, n_jobs=1)
    acc.fit(finalList, classList)
    scoreLR = acc.score(testFinalList, testClassList)
    scoreLRPredict = acc.predict(testFinalList)
    # recall_LR = recall_score(testClassList, scoreLRPredict)
    precision_LR = precision_score(testClassList, scoreLRPredict)
    confusion_matrix_LR = confusion_matrix(testClassList, scoreLRPredict)
    print("Logistic Regression score: ", scoreLR)
    # print("Logistic Regression recall: ", recall_LR)
    print("Logistic Regression: ", precision_LR)
    print("Logistic Regression: ", scoreLR)
    print("Logistic Regression: ", confusion_matrix_LR)


if __name__ == '__main__':
    # help menu
    walk_dir = "/mnt/c/Users/tejamukka/Desktop/genres.tar/genres"

    i = 0.0
    print('walk_dir = ' + walk_dir)
    Y = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
         'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
    # counter_list = list(enumerate(Y, 1))
    finalList = []
    classList = []
    testFinalList = []
    testClassList = []
    for root, subdirs, files in os.walk(walk_dir):
        i = 0
        for filename in files:
            i += 1
            if filename.endswith('.au'):
                file_path = os.path.join(root, filename)
                # print('\t- file %s (full path: %s)' % (filename, file_path))
                ppFileName = rreplace(file_path, ".au", ".pp", 1)

                try:
                    signal, fs = librosa.load(file_path)
                    mfccs = librosa.feature.mfcc(signal, sr=fs, n_mfcc=12)
                    l = mfccs.shape[1]
                    mfccs = mfccs[:, int(0.1 * l):int(0.9 * l)]

                    mean = mfccs.mean(axis=1)
                    covar = np.cov(mfccs, rowvar=1)

                    mean.resize(1, mean.shape[0])
                    # it returns matrix.. not useful for machine learning algorithms except KNN
                    npArray = np.concatenate((mean, covar), axis=0)
                    templist = []
                    for ls in np.nditer(npArray):
                        templist.append(ls)
                    if (i > 70):
                        testFinalList.append(templist)
                        testClassList.append(Y[filename.split('.')[0]])
                    else:
                        finalList.append(templist)
                        classList.append(Y[filename.split('.')[0]])


                        # prepossessingAudio(file_path, ppFileName)
                except Exception as e:
                    print ("Error accured" + str(e))

    # print("Logistic Regression: ", scoreLR*100,"%")

    # MLP
#    classifyPerceptron()

    # SVM
    classfiySVM()

    # KMeans
    

    # Logistic Regression
    classifyLogisticRegression()
    classifyNB()
    classifyPerceptronOnly()
    classifyKMeans()
	
