#Rudd Johnson
#5/28/17

import pandas as pd
import numpy as np
from numpy import random
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import math

#Read in training file optdigits.train into matrix Train
TrainTemp = pd.read_csv('optdigits.train',header=None).as_matrix()
TrainTarget = TrainTemp[:,-1]				#array of train targets
TrainFeature = TrainTemp[:,:-1]				#array of train features

#Read in testing file optdigits.test into matrix Test
TestTemp = pd.read_csv('optdigits.test',header=None).as_matrix()
TestTarget = TestTemp[:,-1]					#array of test targets
TestFeature = TestTemp[:,:-1]				#matrix of test features

#Calculate the euclidian distance
def euclidDistance(x,m):
	return math.sqrt(np.sum((x-m)**2))

#Calculate the Mean Square Error
def MSE(d,u,c):
	summation =0
	for i in range(0,c):
		summation +=euclidDistance(d[i],u)
	return summation/float(c)	  			#average of euclidian distance between centroid and it's elements

#Calculate the Average mean square error for all centroids
def averageMSE(mse,K):
	return (np.sum(mse))/float(K)

#Calcuate average pairwise seperation of each cluster
def MSS(mu,K):
	summation =0

	for x in range(0,len(mu)):				#iterate over all of the centroids such that the distance is found for all but self
		for y in range(x+1, len(mu)):
			summation += math.sqrt(np.sum((mu[x] - mu[y])**2))
	return(summation/(K*(K-1)/float(2)))	#perform mean square seperation calculation and return

#Caluclate the entropy (degree to which cluster consists of objects of a single class)
def entropy(cluster):
	uniqueClasses,countClasses = np.unique(cluster, return_counts=True)	#store each unique classification and it's count
	countClasses = countClasses/float(len(cluster))						#find average occurence of each classification in cluster
	countClasses = -(countClasses * (np.log2(countClasses)))			#evaluate entropy of each value and return sum
	return np.sum(countClasses)

#calculate mean entropy across all clusters
def meanEntropy(clusterClass,K):
	clusterEnt =[]
	totalLength =0
	avg=0																#store the entropy of each cluster
	for i in range(0,len(clusterClass)):
		clusterEnt.append(entropy(clusterClass[i]))
		totalLength += len(clusterClass[i])
	for i in range(0,len(clusterClass)):
		avg+= (len(clusterClass[i])/float(totalLength))*(clusterEnt[i])
	return avg

#Test for convergence of centroids, the condition at which training stops
#return true if previous interations centroids are the same as current
#iterations centroid.
def convergence(prevCentroids,newCentroids):
	if np.array_equal(prevCentroids,newCentroids):	#convergence reached, return true to break training loop
		return True
	return False									#continue training

#main routine which uses above helper function to run k-means clustering
#on training data for 5 training cycle, choosing the best cycle to use for testing

def main():
	K =30										#number of clusters
	mseAvgData = []								#store the avereage MSE for each run
	runClassificationBin = []					#create K bins to store classification of each centroid for
	bestCentroids = []							#hold all centroid values

	for i in range(0,5):						#perform 5 training runs, select the best for testing
		Centroids = random.randint(17,size=(K,64)).astype(dtype=float)	#initialize K centorids to random integers between 0-16
		previousCentroids = None				#store centroid positions from previous iterations
		previousCentroids = np.array(previousCentroids)
		distanceList = np.zeros(shape=(K,3823))   	#create matrix to store distance of instances from each centroid
		mseData =[]	  								#create list of lists to hold MSE for each cluster
		while True:									#iterate until centroids do not change position
			centroidFeatureBin = [[] for b in range(0,K)]	 #create an empty list of list to seperate features
			centroidTargetBin = [[] for c in range(0,K)]	 #create empty list of lists to store corresponding
			for x in range(0,K):			#loop through training features, find the distance between
				for y in range(0,3823):		#data and centroid
					distanceList[x][y] = euclidDistance(TrainFeature[y],Centroids[x])
			for q in range(0,3823):				#iterate over every insance in the training set
				tempBin = []					#create a temporary list to hold the distance between each instance and the cluster
				for r in range(0,K):
					tempBin.append(distanceList[r][q])
				centroidFeatureBin[np.argmin(tempBin)].append(TrainFeature[q]) 	#store clustered feature
				centroidTargetBin[np.argmin(tempBin)].append(TrainTarget[q])	#store clustered labels

			for z in range(0,len(centroidTargetBin)):		#recalculate centroids by finding meann of stored features
				if len(centroidFeatureBin[z])!=0:			#deal with scenario of empty centroid
					Centroids[z] = np.mean(np.array(centroidFeatureBin[z]),axis=0)

			if convergence(previousCentroids,Centroids):		#if convergence has occured, break training loop.
				for x in range(0,len(centroidFeatureBin)):		#iterate all the centroid bins containing features
					if len(centroidFeatureBin[x]) !=0:			#check to make sure the bin actually contains features
						mseData.append(MSE(centroidFeatureBin[x],Centroids[x],len(centroidFeatureBin[x])))#find mean
				runClassificationBin.append(centroidTargetBin) #store the run classifications
				bestCentroids.append(Centroids) 				#append converged centroids
				break

			previousCentroids = np.copy(Centroids)			#set previous set of centroids to new centroids
		mseAvgData.append(averageMSE(mseData,K))			#append
	bestRun = np.argmin(mseAvgData)							#identify the run the lowest MSS
	print "The training stats for best run with K: ",K		#Output chosen test run stats
	print "The avg mean square error is: ",mseAvgData[bestRun]
	print "the MSS is: ", MSS(bestCentroids[bestRun],K)
	print "The mean entropy is: ", meanEntropy(runClassificationBin[bestRun],K)

	chosenCentroids = runClassificationBin[bestRun]			#store chose centroid run class info
	chosenCentroids = np.array(chosenCentroids)				#convert into numpyarray
	centroidClassification = [[] for a in range(0,K)]

	for i in range(0,len(chosenCentroids)):					#identify class associated with each cluster
		if chosenCentroids[i] != []:
			num = np.bincount(chosenCentroids[i])
			centroidClassification[i]=np.argmax(num)

	testDistanceList = np.zeros(shape=(len(bestCentroids[bestRun]),1797))   #create matrix to measure distance test and centroid


	for x in range(0,len(bestCentroids[bestRun])):  		      			 #loop through test features, find the distance
		for y in range(0,1797):
			testDistanceList[x][y] = euclidDistance(TestFeature[y],bestCentroids[bestRun][x])

	testTargetBin = []					#create empty list of lists to store class prediction
	for q in range(0,1797):				#iterate over every insance in the test distance matrix
		tempBin = []					#create a temporary list to hold the distance between each instance and the cluster
		for r in range(0,len(bestCentroids[bestRun])):
			tempBin.append(testDistanceList[r][q])
		testTargetBin.append(centroidClassification[np.argmin(tempBin)])
	print accuracy_score(TestTarget,testTargetBin) 		#outpout the accuracy
	print(confusion_matrix(TestTarget,testTargetBin))	#output confusion matrix

	for i in range(len(bestCentroids[bestRun])):		#output visualization of each cluster center to file
    		plt.clf()
   		pic= np.reshape(bestCentroids[bestRun][i].astype(int), (8, 8))
   		plt.imshow(pic, cmap='gray')
		plt.savefig(str(i))
main()
