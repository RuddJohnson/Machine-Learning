'''
Rudd Johnson
4/14/17
'''
import random
import numpy as np
class Perceptron:								#perceptron class
	def __init__(self,target=0,n=0):
		self.target = float(target)/255			#store perceptron class value (0-9)
		self.weight = []
		for x in range(0,785):
			self.weight.append(random.uniform(-0.05,0.05))		#list of weights for all 784 inputs
		self.n=n							#learning rate
		self.weightInputSum=0.0 			#summation changes with each input set, used for accuracy calculation

	def activation(self,row):				#activation function determines if perceptron fires (1) or not (0)
		self.weightInputSum = 1 * self.weight[0]	#bias
		temp = row[0]					#store correct integer of row
		row[0] = 1.0					#bias input
		self.weightInputSum = self.weightInputSum + np.dot(row,self.weight)	#dot product input and weight
		row[0] = temp					#reset correct interger of row as first element in list

	def adjustWeight(self,row):				#training function, adjust weight for each perceptron with each row
		temp = row[0]
		row[0] = 1.0						#store correct interger of row and set first index to bias to allign lists
		if self.weightInputSum >0:			#if perceptron fires, y =1
			for i in range (0,785):
				if temp == self.target:		#if perceptron correctly predicts t =1
					self.weight[i] = self.weight[i] + (self.n*(1 - 1)*row[i])
				else:
					self.weight[i] = self.weight[i] + (self.n*(0 - 1)*row[i])	#else t=0

		else:								#if perceptron doesnt fire y =0
			for i in range (0,785):
				if temp == self.target:		#if perceptron correctly predicts t=1
					self.weight[i] = self.weight[i] + (self.n*(1 - 0)*row[i])
				else:						#else t =0
					self.weight[i] = self.weight[i] + (self.n*(0 - 0)*row[i])
		row[0] = temp

class manage:								#class that manages perceptrons in training and testing
	def __init__(self,n):
		self.testFileData= []				#list of rows in csv test file
		self.trainFileData= []				#list of rows in csv rain file
		self.percep = []					#list of perceptrons
		for i in range(0,10):				#create list of ten perceptrons, initatialzie targets to 0-9, weights initialized -0.5-0.5
			self.percep.append(Perceptron(i,n))

	def readIn(self,trainFileName,testFileName):
		with open(trainFileName,'r') as fileHandle:		#open training file, attach fileHandle to file
			for line in fileHandle.readlines():			#iterate over every row in the file
				line = line.split(',')					#eliminate commas and create index for exery number
				for i in range(0,785):					#iterate over every index in row after targe to format so that:
					line[i] = float(line[i])			#each value is converted float
					line[i] = line[i]/255				#each data value is divided by 255
				self.trainFileData.append(line)			#append row to list of rows
		fileHandle.close()								#close file

		with open(testFileName,'r') as fileHandle:		#open test file, attach fileHandle to file
			for line in fileHandle.readlines():			#iterate over every row in the file
				line = line.split(',')					#eliminate commas and create index for exery number
				for i in range(0,785):					#iterate over every index in row after targe to format so that:
					line[i] = float(line[i])			#each value is converted float
					line[i] = line[i]/255				#each data value is divided by 255
				self.testFileData.append(line)			#append row to list of rows
		fileHandle.close()

	def train(self):									#training function
		correct = 0;									#correct number of predictions
		for k in range(0,60000):						#iterate over all input rows
			highestSum =-100000000.0					#initialze value to hold highest dot product (for predicton)
			predictedVal=0.0;							#store value of class that predicts
			for j in range(0,10):						#for each perceptron:
				self.percep[j].activation(self.trainFileData[k])	#call activation function
				self.percep[j].adjustWeight(self.trainFileData[k])	#adjust weigh based on result
				if highestSum < self.percep[j].weightInputSum:
					highestSum = self.percep[j].weightInputSum		#find output with highest dot product
					predictedVal = self.percep[j].target
			if self.trainFileData[k][0] *255 == predictedVal*255:		#if that turns out to be correct value:
				correct = correct+1										#increment and return correct
		return correct

	def test(self):										#test function idenical to train except no weight adjustment
		correct = 0;									#correct number of predictions
		for k in range(0,10000):						#iterate over all input rows
			highestSum =-100000000.0					#initialze value to hold highest dot product (for predicton)
			predictedVal=0.0;							#store value of class that predicts
			for j in range(0,10):						#for each perceptron:
				self.percep[j].activation(self.testFileData[k])			#call activation function
				if highestSum < self.percep[j].weightInputSum:
					highestSum = self.percep[j].weightInputSum			#find predicting class
					predictedVal = self.percep[j].target
			if self.testFileData[k][0] *255 == predictedVal*255:		#if that is correct value:
				correct = correct+1										#incrememnt and return correct
		return correct;

	def epoch0Train(self):							#run training data once without adjusting weights
		correct = 0;
		for k in range(0,60000):					#iterate over all rows
			highestSum =-100000000.0				#initialize varivable that stores highes sum
			predictedVal=0.0;						#holds predicted class value
			for j in range(0,10):					#iterate over all perceptrons
				self.percep[j].activation(self.trainFileData[k])	#fun activation function
				if highestSum < self.percep[j].weightInputSum:		#find predicting class
					highestSum = self.percep[j].weightInputSum
					predictedVal = self.percep[j].target
			if self.trainFileData[k][0] *255 == predictedVal*255:		#if predicted value is correc
				correct = correct+1										#increment and return correct
		return correct;


	def confuseMatrix(self,confusion):		#build confusion matric, goes over test one last time after last round of training
		for k in range(0,10000):
			highestSum =-100000000.0		#call activation function and store predicted output same way as test
			predictedVal=0.0;
			for j in range(0,10):
				self.percep[j].activation(self.testFileData[k])
				if highestSum < self.percep[j].weightInputSum:
					highestSum = self.percep[j].weightInputSum
					predictedVal = self.percep[j].target
			predict =  predictedVal*255
			predict = int(predict)							#recast predicted value as int from float
			predictClass = self.testFileData[k][0] * 255	#store predicted class recast as int from float
			predictClass = int(predictClass)
			confusion[predictClass][predict] = confusion[predictClass][predict]+ 1	#go to that index in matrix, increment once

trainResults = []					#store results from each training epoch
testResults = []					#store results from each testing epoch
confusion = np.zeros((10,10))		#make 10 x 10 confusion matrix

nVal=input("enter learning rate: ")			#prompt user for learning rate and file name then echo
nVal = float(nVal)
filename = input("please enter filename: ")
print("learning rate is: ",nVal)
print("filename is: ",filename)

learning =manage(nVal)								#instantiate manager class
learning.readIn("mnist_train.csv","mnist_test.csv")	#pass files to read in function
out = open(filename,"a")							#open output file

correct = learning.epoch0Train()					#intially call epoch 0 training function, which doesnt adjust weights
trainResults.append((float(correct)/60000)*100)		#append results with accuracy determined
correct = learning.test()							#call epoch 0 test, append results with accuracy determined
testResults.append((float(correct)/10000)*100)

for i in range(1,50):						#next 49 epochs
	print("epoch ",i)
	correct = learning.train()				#call raininbg function with weight adjustment and store accuracy
	trainResults.append((float(correct)/60000)*100)
	print(correct)
	correct = learning.test()				#call testing and store accuracy
	testResults.append((float(correct)/10000)*100)
	print(correct)

print("confusion matrix")
learning.confuseMatrix(confusion)					#create confusion matrix

out.write("training")								#output file formatting:
out.write("\n")
for i in range(0,len(trainResults)):				#output trianing accuracy
	out.write(str(trainResults[i]))
	out.write("\n")

out.write("testing")
out.write("\n")										#output testing accuracy
for i in range(0,len(testResults)):
	out.write(str(testResults[i]))
	out.write("\n")
out.write("\n")


for i in range(0,10):						#ouput confusion matrix
	for j in range(0,10):
		out.write(str(confusion[i][j]))
		out.write(" ")
	out.write("\n")

out.close()							#close file
