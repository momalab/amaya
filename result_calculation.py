from os import listdir
from os.path import isfile, join
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC

# CHANGE THIS ------------------ *********** >>>>>>>>>>>>>>
FEATUREDIRPATH = 'dataset/x86_64/'

def PerformanceMeasure(yTest, yPredict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(yPredict)):
        if yTest[i]==yPredict[i]==1:
           TP += 1
        if yPredict[i]==1 and yTest[i]!=yPredict[i]:
           FP += 1
        if yTest[i]==yPredict[i]==0:
           TN += 1
        if yPredict[i]==0 and yTest[i]!=yPredict[i]:
           FN += 1

    return(TP, FP, TN, FN)


if __name__ == '__main__':
	fileNameList = [f for f in listdir(FEATUREDIRPATH) if isfile(join(FEATUREDIRPATH, f))]
	downSampleRateList = []

	for fileName in fileNameList:
		downSampleRateList.append(fileName.split('_')[-1].split('.')[0])

	# Sort the downsampling rates
	downSampleRateList = [int(i) for i in downSampleRateList]
	downSampleRateList.sort()
	downSampleRateList = [str(i) for i in downSampleRateList]

	for downSampleRate in downSampleRateList:
		featureFileName = 'features_' + downSampleRate + '.txt'
		featuresDF = pd.read_csv(FEATUREDIRPATH + featureFileName, index_col=0)
		trainDF, testDF = train_test_split(featuresDF, test_size=0.2, shuffle=False)

		hashTrain = trainDF.loc[:, : '0'].to_numpy()
		hashTest = testDF.loc[:, : '0'].to_numpy()

		xTrain = trainDF.to_numpy()[:, 1:-1]
		xTest = testDF.to_numpy()[:, 1:-1]

		yTrain = trainDF.to_numpy()[:, -1:]
		yTest = testDF.to_numpy()[:, -1:]

		np.savetxt(FEATUREDIRPATH + 'hashTrain.txt', hashTrain, fmt='%s', delimiter=',')
		np.savetxt(FEATUREDIRPATH + 'hashTest.txt', hashTest, fmt='%s', delimiter=',')
		np.savetxt(FEATUREDIRPATH + 'xTrain.txt', xTrain, fmt='%s', delimiter=',')
		np.savetxt(FEATUREDIRPATH + 'yTrain.txt', yTrain, fmt='%s', delimiter=',')
		np.savetxt(FEATUREDIRPATH + 'xTest.txt', xTest, fmt='%s', delimiter=',')
		np.savetxt(FEATUREDIRPATH + 'yTest.txt', yTest, fmt='%s', delimiter=',')

		xTrain = np.loadtxt(FEATUREDIRPATH + 'xTrain.txt', delimiter=',')
		xTest = np.loadtxt(FEATUREDIRPATH + 'xTest.txt', delimiter=',')
		yTrain = np.loadtxt(FEATUREDIRPATH + 'yTrain.txt', delimiter=',')
		yTest = np.loadtxt(FEATUREDIRPATH + 'yTest.txt', delimiter=',')

		standardScaler = preprocessing.StandardScaler()
		xTrain = standardScaler.fit_transform(xTrain)
		xTest = standardScaler.transform(xTest)

		clf = SVC(gamma='auto', kernel='rbf')
		clf.fit(xTrain, yTrain)
		yPredict = clf.predict(xTest)
		accuracy = np.mean(yPredict == yTest)

		TP, FP, TN, FN = PerformanceMeasure(yTest, yPredict)
		print("Accuracy with SVM Standard Scalar (Sampling Rate: %s): %f \t TP: %i \t FP: %i \t TN: %i \t FN: %i"%(downSampleRate, accuracy, TP, FP, TN, FN))

		os.remove(FEATUREDIRPATH + 'hashTrain.txt')
		os.remove(FEATUREDIRPATH + 'hashTest.txt')
		os.remove(FEATUREDIRPATH + 'xTrain.txt')
		os.remove(FEATUREDIRPATH + 'yTrain.txt')
		os.remove(FEATUREDIRPATH + 'xTest.txt')
		os.remove(FEATUREDIRPATH + 'yTest.txt')