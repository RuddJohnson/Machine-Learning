#Rudd Johnson
#5/8/17

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

class manager:									#class manages the data preprocessing for test and training
	def __init__(self):
		self.testData= []						#list of rows in data test file
		self.trainData= []						#list of rows in data rain file
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

		trainMean = self.trainFeatures.mean(axis=0)	#find the mean of training features
		trainSD = self.trainFeatures.std(axis =0)	#find the standard deviation of training features

		self.trainFeatures = preprocessing.scale(self.trainFeatures)	#preprocess training features (subtract mean and divide by SD)
		self.testFeatures = (self.testFeatures - trainMean)/trainSD		#preprocess test features with training mean and SD

def experiment1(trainFeatures, trainLabel, testFeatures, testLabel):
	classifier = SVC(kernel ="linear", C=1, probability = True)		#create SVC
	classifier.fit(trainFeatures,trainLabel)						#train classifier with training features

	testPredict =classifier.predict(testFeatures) 					#predict test features with trained SVM
	testPredict = np.array(testPredict)								#turn into numpy array
	testPrec = metrics.precision_score(testLabel ,testPredict )		#find precision, accuracy, and recall of predictions
	testAcc = metrics.accuracy_score(testLabel ,testPredict )
	testRec = metrics.recall_score(testLabel ,testPredict )

	print"Accuracy: ",testAcc * 100									#output precision, accuracy, and recall
    	print"Precision: ",testPrec * 100
    	print"Recall: ",testRec * 100

	testPredict =classifier.predict_proba(testFeatures)				#find prediction probability
    	falsePos, truePos,threshholds = metrics.roc_curve(testLabel,np.ravel(testPredict[:,-1])) # create ROC curce
	plt.figure()													#plot ROC cuve
    	lw = 2
    	plt.plot(falsePos, truePos, color='darkorange',lw=lw)
    	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    	plt.xlim([0.0, 1.0])                   				 #range
    	plt.ylim([0.0, 1.05])
    	plt.xlabel('False Positive Rate')       			#x axis label
    	plt.ylabel('True Positive Rate')       				#y axis label
    	plt.title('Experiment 1')							#title
    	plt.savefig('Experiment1.png', bbox_inches='tight')	#save figure to file
    	plt.show()										    #output

	return classifier 										#return the SVM

def experiment2(trainFeatures, trainLabel, testFeatures, testLabel,classifier):
    	accuracy = []    					 				#store accuracy
    	weights = np.argsort(classifier.coef_[0])  		 	#sort in descending order
	weights = weights[::-1]                			 		#invert into ascending order
	print(weights)						 					#output the weights for finding most heavily weighted features
	tempTrain = trainFeatures[:,weights]					#create temp list of those training features
	tempTest = testFeatures[:,weights]			 			#create temp list of those test features
	tempFeature = SVC(kernel='linear', C=1, probability=True)	#create temp SVM
	for i in range(1,57):      				 					#iterate over features
        	tempFeature.fit(tempTrain[:,:i],trainLabel)     	#train with different features
		accuracy.append(tempFeature.score(tempTest[:,:i], testLabel)) #Record the accuracy

   	m = np.arange(1,57)											 #56 or the features plotted
    	plt.figure()
    	lw = 2
    	plt.plot(m, accuracy, color='darkorange',lw=lw)			 #plot accuracy vs number of features
    	plt.xlim([0.0, 58])                    					 #ranges
    	plt.ylim([0.0, 1.0])
    	plt.xlabel('Number of Features')       					 #x axis label
    	plt.ylabel('Accuracy')                 					 #y axis label
    	plt.title('Experiment 2')	       	   					 #title of experiment
    	plt.savefig('Experiment2.png', bbox_inches='tight')		 #save image to file
    	plt.show()  			      							 #display graph

def experiment3(trainFeatures, trainLabel, testFeatures, testLabel, classifier):
	accuracy = []    					 						#store accuracy
	tempFeature = SVC(kernel='linear', C=1, probability=True)	#create temporary SVM to train
	for i in range(2,58):      									#iterate over features
		features = np.random.choice(57, size =i, replace =False)        #randomly choose number without replacement
		tempFeature.fit(trainFeatures[:,features],trainLabel)   		#train SVM with randomly selected features
        	accuracy.append(tempFeature.score(testFeatures[:,features], testLabel)) #Record the accuracy

   	m = np.arange(2,58)						#same implementation as the graph in experiment 2
    	plt.figure()
    	lw = 2
    	plt.plot(m, accuracy, color='darkorange',lw=lw)		 #plot accuracy vs number of features
    	plt.xlim([0.0, 58])                 			 	 #ranges
    	plt.ylim([0.0, 1.0])
    	plt.xlabel('Number of Randomly Selected Features')   #x axis label
    	plt.ylabel('Accuracy')              			 	 #y axis label
    	plt.title('Experiment 3')
    	plt.savefig('Experiment3.png', bbox_inches='tight')	 #save to file
    	plt.show()  						 				 #display graph

def main():													#entry point, preprocess data and call experiments 1-3
	SVM = manager()
	clf = experiment1(SVM.trainFeatures,SVM.trainLabel,SVM.testFeatures,SVM.testLabel)
	experiment2(SVM.trainFeatures,SVM.trainLabel,SVM.testFeatures,SVM.testLabel,clf)
	experiment3(SVM.trainFeatures,SVM.trainLabel,SVM.testFeatures,SVM.testLabel,clf)
main()
