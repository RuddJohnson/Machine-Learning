#Rudd Johnson
#Assignment 4

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import math
class manager:									#class manages the data preprocessing for test and training
	def __init__(self):
		self.testData= []						#list of rows in data test file
		self.trainData= []						#list of rows in data rain file
		self.probPositive =0.0;					#store the probability of spam in training set
		self.probNegative =0.0;					#store pron of not spam in training set
		self.confusion = np.zeros((2,2))		#make 10 x 10 confusion matrix
		#read in traing file, and test file, note these were shuffled and evenly split using shuff() and split -l 2300
		TrainTemp = pd.read_csv("training.data",sep = ',',dtype= 'float64', header = None).as_matrix()
		TestTemp = pd.read_csv("testing.data",sep = ',',dtype= 'float64', header = None).as_matrix()

		#append to respective data to respective listst
		for i in range(len(TrainTemp)):
			self.trainData.append(TrainTemp[i])
		for i in range(len(TestTemp)):
			self.testData.append(TestTemp[i])

		self.testData = np.array(self.testData)		#convert data from lists to numpy arrays
		self.trainData = np.array(self.trainData)

		self.trainFeatures = self.trainData[:, :-1]	#seperate training features from training label
    		self.trainLabel = np.ravel(self.trainData[:, -1])

		self.testFeatures = self.testData[:, :-1]	#seperate test features from test label
    		self.testLabel = np.ravel(self.testData[:, -1])

		pos = neg = 0								#local variables hold number spam versus not spam
		for i in range(0,self.trainLabel.size):		#iterate over training labels which hold classification for training data
			if self.trainLabel[i] == 1:				#if it is spam, increment positive
				pos = pos +1
			else:
				neg = neg+1									#if it isn't spam, increment negative
		self.probPositive = float(pos)/self.trainLabel.size	#find probability of spam vs not spam in training data
		self.probNegative = float(neg)/self.trainLabel.size
		print self.probPositive								#output probability of spam in trianing data
		print self.probNegative

		#split the training data into two different arrays, one for positve classification and one for negative classification
		trainPositive = []
		trainNegative = []
		for i in range(0,self.trainLabel.size):		#iterate over trainLabels that correspond to training data instances
			if self.trainLabel[i] == 1:				#if it is positive, add that row of features to trainPositive
				trainPositive.append(self.trainFeatures[i])
			else:									#if it is negative, add that row of features to trainNegative
				trainNegative.append(self.trainFeatures[i])
		trainPositive = np.array(trainPositive)		#convert lists to numpy arrays
		trainNegative = np.array(trainNegative)

		meanPos = np.mean(trainPositive,axis =0)	#calculate mean and standard deviation for each
		meanNeg = np.mean(trainNegative, axis=0)
		stdPos = np.std(trainPositive,axis =0)
		stdNeg = np.std(trainNegative, axis=0)
		stdPos[stdPos==0.0]=0.0001					#convert zero to 0.0001 to avoid divide  by zero
		stdNeg[stdNeg==0.0]=0.0001
		#running naive bayes on test set
		predict = []
		for i in range(0,2300):						#iterate over every test instance and run probability density formula
			posProbFunc = np.log((1/(math.sqrt(2*math.pi)*stdPos))*np.exp(-((self.testFeatures[i]-meanPos)**2)/(2*stdPos**2)))
			negProbFunc = np.log((1/(math.sqrt(2*math.pi)*stdNeg))*np.exp(-((self.testFeatures[i]-meanNeg)**2)/(2*stdNeg**2)))
			posClass = np.log(self.probPositive) + np.sum(posProbFunc)	#sum the logs of probability density formula
			negClass = np.log(self.probNegative) + np.sum(negProbFunc)	#to prevent underflow
			if posClass > negClass:					#if the positive class is great, append 1 else 0
				predict.append(1)
			else:
				predict.append(0)
		self.accuracy = str(metrics.accuracy_score(self.testLabel,predict))		#determine accuracy
		self.precision = str(metrics.precision_score(self.testLabel,predict))	#determine precision
		self.recall = str(metrics.recall_score(self.testLabel,predict))			#determine recall
		for i in range(0,self.testLabel.size):									#create confusion matrix
			self.confusion[int(self.testLabel[i])][int(predict[i])] = self.confusion[int(self.testLabel[i])][int(predict[i])]+1
		print "Accuracy: ",self.accuracy 										#output accuracy, precision, and recall
		print "Pecision: ",self.precision
		print "recall: ",self.recall
		print (self.confusion)


def main():		#entry point
	bayes = manager()

main()
