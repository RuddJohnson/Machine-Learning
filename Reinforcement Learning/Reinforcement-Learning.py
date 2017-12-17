#Rudd Johnson
#6/1/17

import numpy as np
from random import randint
from enum import Enum
import itertools

#This class is responsible for holding holding all the
#pariculars of the reinforcement learning program. More specifically,
#The q-matrix and training as well as testing algorithms are in this class
class Reinforce:

	def __init__(self):								#constructor
		self.robbyLoc =[]
		self.state = ""								#store the string holding robby's state
		self.score =[]								#list of scores
		self.puzzle=None							#puzzle to hold robby and his cans
		self.Qtable = {}							#create dictionary of states
		self.testScore =[]
		outcomes = ('W','E','C')					#tuple of outcome
		total = [''.join(p) for p in itertools.product(outcomes,repeat=5)]	#create evert permutation of every possible state, 3^5
		self.Qtable = {i: [0 for x in range(5)] for i in total}				#add all possible permutations to dictionary
		self.n =0.2
		self.Epsilon =1
		self.gamma = 0.9

	#fill 10 x 10 puzzle with random cans and place robby at random location
	def setPuzzle(self):										#reset the board and robby
		self.puzzle=np.random.choice([0,1],size=(10,10),replace=True, p=[0.5,0.5])
		self.robbyLoc =[]
		for i in range(2):
			self.robbyLoc.append(np.random.choice(range(0,10),replace =True))
		self.robbyLoc = np.array(self.robbyLoc)					#Robby's location in the board

	#order of state is NSEWH, will take a certain state
	#and determine what W,C,or E are NSEWH of him. This is the
	#mapped to Q table
	def evaluateState(self,Puzzle):
		#print "State in evaluate state: ",self.state
		#print "Robby location: ",self.robbyLoc
		#print self.puzzle
		state = ""
		Y = self.robbyLoc[1]
		X= self.robbyLoc[0]
		if X == 0:						#at north wall
			state = "W"
		if X>0:
			if Puzzle[X-1][Y] == 1:		#can north
				state = "C"
			else:
				state = "E"				#empty north

		#check south
		if X == 9:						#at south wall
			state += "W"
		if X<9:
			if Puzzle[X+1][Y] == 1:		#can south
				state +="C"
			else:
				state +="E"				#empty south

		#check east
		if Y == 9:						#at east wall
			state +="W"
		if Y <9:
			if Puzzle[X][Y+1] == 1:		#can east
				state += "C"
			else:
				state += "E"			#empty east

		#check west
		if Y == 0:						#at west wall
			state += "W"
		if Y >0:
			if Puzzle[X][Y-1] == 1:		#can west
				state += "C"
			else:
				state +="E"				#empty west

		#pick up
		if Puzzle[X][Y]	== 1:			#can at pick up
			state += "C"
		else:
			state += "E"				#no can where standing

		self.state = state				#update state
		return state

	#function associates items in the direction that robby is trying to go
	#a temp score is adjusted and robby is moved appropriately. This temp
	#score is added to the total score for the episode
	def moveRobby(self,selection):
		if selection == 0:				#north
			if self.state[0] == "W":	#if wall, return negative 5, do not update position
				 return -5
			self.robbyLoc[0]-=1			#increment north by 1 and return 0
			self.evaluateState(self.puzzle)	#update state
			return 0
		elif selection ==1:				#south
			if self.state[1] == "W":
				return -5
			self.robbyLoc[0]+=1			#decrement Y by 1
			self.evaluateState(self.puzzle)	#update state
			return 0
		elif selection ==2:				#east
			if self.state[2] == "W":
				return -5				#decrement by 5 for running into wall
			self.robbyLoc[1]+=1			#move east
			self.evaluateState(self.puzzle)	#update state
			return 0
		elif selection ==3:				#west
			if self.state[3] == "W":
				return -5				#decrement by 5 for running into wall
			self.robbyLoc[1]-=1			#move robby west
			self.evaluateState(self.puzzle)	#update state
			return 0
		elif selection ==4:				#pick up
			if self.state[4] == "E":    #if space is empty return -1
				return -1				#decrement for trying to pick up can that isn't there
			self.puzzle[self.robbyLoc[0]][self.robbyLoc[1]]=0	#remove can from location in puzzle
			self.evaluateState(self.puzzle)	#update state
			return 10					#picked up can

	#returns the action from a given sate in a Q state with the hights action value
	def greedyPick(self):
		#break tie if all actions equal
		if self.Qtable[self.state][0]==self.Qtable[self.state][1]==self.Qtable[self.state][2]==self.Qtable[self.state][3]==self.Qtable[self.state][4]:
			return randint(0,4)			#if all states equal, return random state to break tie
		return np.argmax(self.Qtable[self.state])

	#function updates the q table after each move
	def updateQtable(self,oldState,newState,action,reward):
		maxAction = max(self.Qtable[newState])			#store the max action from the next state
		Qori = self.Qtable[oldState][action]			#stoe Q value from starting state
		self.Qtable[oldState][action] = Qori + self.n*(reward+self.gamma*maxAction -Qori)	#update

	#train robby over 5000 episodes with 200 moves per epsode. Using greedy epsilon
	#algorithm to determine which action to take, decrementing epsilon by 1 every 50 episodes by 0.01 (starts at 1)
	#until epsilon of 0.1 is reached
	def train(self):
		for i in range(0,5000):
			if i % 50 == 0 and self.Epsilon >0.1:			#decrement epsilon by 0.01 every 50 episodes
				self.Epsilon -=0.01
			tempScore = 0									#initialize score to 0
			self.setPuzzle()								#reset puzzle after each episode
			self.state = self.evaluateState(self.puzzle)	#find the starting state
			for x in range (0,200):							#200 moves per episode
				if np.random.choice([0, 1], p =[self.Epsilon, 1-self.Epsilon]):#epsilon greedy algorithm
					nextMove = self.greedyPick()			#greedy
				else:
					nextMove = randint(0,4)					#pick random move
				oldState = self.state						#previous robby state
				reward = self.moveRobby(nextMove)    		#reward for robby's movement
				tempScore +=reward
															#tempScore -=0.5
				self.updateQtable(oldState,self.state,nextMove,reward)
			if i % 100 == 0:								#add cumulative score every hundred episodes
					self.score.append(tempScore)

	#test robby after he has been trained with epsilon of 0
	#do identical to trianing process except epsilon remains zero and
	#Qtable is not updated
	def test(self):
		for i in range (0,5000):
			tempScore = 0									#initialize score to 0
			self.setPuzzle()								#reset puzzle after each episode
			self.state = self.evaluateState(self.puzzle)	#find the starting state
			for x in range(0,200):
				nextMove = self.greedyPick()				#picky max action
				oldState = self.state						#previous robby state
				reward = self.moveRobby(nextMove)			#reward for robby's movement
				tempScore +=reward
															#tempScore -=0.5
			if i % 100 == 0:								#add cumulative score every hundred episodes
				self.testScore.append(tempScore)
#Main routine, entry point of program
def Main():
	learn = Reinforce()				#make instance of class
	learn.train()					#train robby
	print "Training Reward Plot"	#output rewards plot
	for i in learn.score:
		print i

	print learn.score				#test robby
	learn.test()					#output rewards plot
	print"Testing Reward Plot"
	for i in learn.testScore:
		print i
	learn.testScore = np.array(learn.testScore)	#output standard deviation and mean of test
	print "Average awards over test episodes: "
	print np.average(learn.testScore)
	print "Standard deviation: "
	print np.std(learn.testScore)

Main()
