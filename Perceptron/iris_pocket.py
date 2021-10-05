import random 
import numpy as np

#Initializing the different Iris classes to binary values (put binary value in an array for simplicity)
IrisSetosa = [0,0,0]
IrisVersicolor = [1,1,0]
IrisVirginica = [1,1,1]

#initialize epoch to zero - will limit number of epochs incase data isn't linearly seperable
epoch = 0

#open training dataset and split each training data point 
IrisTrainData = open("iris_train.txt", "r")
IrisTrainDataLines = IrisTrainData.readlines() 

#split each dimension of input data for each input into an array and convert floats in strings to actual floats
for x in range(len(IrisTrainDataLines)):
	IrisTrainDataLines[x] = IrisTrainDataLines[x].split(",")
	for y in range(4):
		IrisTrainDataLines[x][y] = float(IrisTrainDataLines[x][y])

#converting ground truth labels from strings to binary
for x in range(40):
	IrisTrainDataLines[x][4] = IrisSetosa
	
for x in range(40,80):
	IrisTrainDataLines[x][4] = IrisVersicolor

for x in range(80,120):
	IrisTrainDataLines[x][4] = IrisVirginica

#initialize learning rate (chose value between 0.1 and 0.9)
c = 0.9

#generate matrix with random floats ranging from -1 to 1 and absorb bias (w0 included in matrix)
w_matrix = np.random.uniform(low = -1, high = 1, size = (5,3))

#initialize pocket of size 2 to hold best weight vector and best run length for each output node
pocket = [w_matrix, 0] 

#initialize variable to hold number of correct classifications in a row
correct = 0

#TRAINING
while (epoch != 100):
	for p in range(len(IrisTrainDataLines)):
		#extract input data from test point p and preprocess it by removing ground truth label and absorbing the bias
		IrisTrainMinusTruthLabel = IrisTrainDataLines[p][0:4]
		#bias absorbed so x0 = 1
		IrisTrainMinusTruthLabel.insert(0,1)
		
		#calculate actual output for outputs nodes 1 and 2
		y_1 = np.sum([a * b for a, b in zip(w_matrix[:,0],IrisTrainMinusTruthLabel)])
		y_2 = np.sum([a * b for a, b in zip(w_matrix[:,1],IrisTrainMinusTruthLabel)])
		y_3 = np.sum([a * b for a, b in zip(w_matrix[:,2],IrisTrainMinusTruthLabel)])
		
		#set output of both output nodes equal to either 1 or 0 depending on sign of actual output
		if y_1 >= 0:
			y_1 = 1
		elif y_1 < 0:
			y_1 = 0
			
		if y_2 >= 0:
			y_2 = 1
		elif y_2 < 0:
			y_2 = 0
			
		if y_3 >= 0:
			y_3 = 1
		elif y_3 < 0:
			y_3 = 0	
		
		#change in weight to be added/subtracted from current weight		
		delta = [i * c for i in IrisTrainMinusTruthLabel]	
			
		#if classified correctely, increment count. If misclassified, compare pocket with current count, and then adjust weight matrix	
		if y_1 == IrisTrainDataLines[p][4][0] and y_2 == IrisTrainDataLines[p][4][1] and y_3 == IrisTrainDataLines[p][4][2]:
			correct += 1
		elif y_1 != IrisTrainDataLines[p][4][0] or y_2 != IrisTrainDataLines[p][4][1] or y_3 != IrisTrainDataLines[p][4][2]:
			if correct > pocket[1]:
				pocket = (w_matrix, correct)
			else:
				pocket = pocket
				
			#changing weights attached to each output depending on the direction of the difference between the output of output nodes and ground truth labels
			if y_1 > IrisTrainDataLines[p][4][0]:
				w_matrix[:,0] = w_matrix[:,0] - delta
			elif y_1 < IrisTrainDataLines[p][4][0]:
				w_matrix[:,0] = w_matrix[:,0] + delta
			if y_2 > IrisTrainDataLines[p][4][1]:
				w_matrix[:,1] = w_matrix[:,1] - delta
			elif y_2 < IrisTrainDataLines[p][4][1]:
				w_matrix[:,1] = w_matrix[:,1] + delta
			if y_3 > IrisTrainDataLines[p][4][2]:
				w_matrix[:,2] = w_matrix[:,2] - delta
			elif y_3 < IrisTrainDataLines[p][4][2]:
				w_matrix[:,2] = w_matrix[:,2] + delta
				
			#reset correct count to zero if a misclassification occurs
			correct = 0
				
	epoch += 1
	
	#after each epoch, update weight matrix with weight matrix in the pocket
	w_matrix = pocket[0]
	
	#reset correct count to zero for next epoch
	correct = 0
	
#ACCURACY CALCULATION (CREATING CONFUSION MATRIX)
#open testing dataset and split each testing data point 
IrisTestData = open("iris_test.txt", "r")
IrisTestDataLines = IrisTestData.readlines() 

#split each dimension of input data for each input into an array and convert floats in strings to actual floats
for x in range(len(IrisTestDataLines)):
	IrisTestDataLines[x] = IrisTestDataLines[x].split(",")
	for y in range(4):
		IrisTestDataLines[x][y] = float(IrisTestDataLines[x][y])
		
#converting ground truth labels from strings to binary
for x in range(10):
	IrisTestDataLines[x][4] = IrisSetosa
	
for x in range(10,20):
	IrisTestDataLines[x][4] = IrisVersicolor

for x in range(20,30):
	IrisTestDataLines[x][4] = IrisVirginica

CorrectOutputClassSetosa = 0
CorrectOutputClassVersicolor = 0
CorrectOutputClassVirginica = 0

MisclassOfSetosaForVersicolor = 0
MisclassOfSetosaForVerginica = 0
MisclassOfVersicolorForSetosa = 0
MisclassOfVersicolorForVerginica = 0
MisclassOfVerginicaForSetosa = 0
MisclassOfVerginicaForVersicolor = 0

#iterate through each of the test data to determine their class labels
for x in range(len(IrisTestDataLines)):
	#extract input data from test point p and preprocess it by removing ground truth label and absorbing the bias
	IrisTestMinusTruthLabel = IrisTestDataLines[x][0:4]
	#bias absorbed so x0 = 1
	IrisTestMinusTruthLabel.insert(0,1)	

	#calculate actual output for outputs nodes 1,2 and 3
	y_1 = np.sum([a * b for a, b in zip(w_matrix[:,0],IrisTestMinusTruthLabel)])
	y_2 = np.sum([a * b for a, b in zip(w_matrix[:,1],IrisTestMinusTruthLabel)])
	y_3 = np.sum([a * b for a, b in zip(w_matrix[:,2],IrisTestMinusTruthLabel)])
	
	#set outout of both output nodes equal to either 1 or 0 depending on sign of actual output
	if y_1 >= 0:
		y_1 = 1
	elif y_1 < 0:
		y_1 = 0
		
	if y_2 >= 0:
		y_2 = 1
	elif y_2 < 0:
		y_2 = 0
		
	if y_3 >= 0:
		y_3 = 1
	elif y_3 < 0:
		y_3 = 0
		
	#to determine diagonal of confusion matrix (basically where y(x) = d), need to sum correct classifications per class
	if (y_1 == 0 and IrisTestDataLines[x][4][0] == 0) and (y_2 == 0 and IrisTestDataLines[x][4][1] == 0) and (y_3 == 0 and IrisTestDataLines[x][4][2] == 0):
		CorrectOutputClassSetosa += 1
	elif (y_1 == 1 and IrisTestDataLines[x][4][0] == 1) and (y_2 == 1 and IrisTestDataLines[x][4][1] == 1) and (y_3 == 0 and IrisTestDataLines[x][4][2] == 0):
		CorrectOutputClassVersicolor += 1
	elif (y_1 == 1 and IrisTestDataLines[x][4][0] == 1) and (y_2 == 1 and IrisTestDataLines[x][4][1] == 1) and (y_3 == 1 and IrisTestDataLines[x][4][2] == 1):
		CorrectOutputClassVirginica += 1
	
	#to determine the rest of the confusion matrix, need to determine how many misclassifications are occuring for each possible combination of classes
	if IrisTestDataLines[x][4][0] == 0 and IrisTestDataLines[x][4][1] == 0 and IrisTestDataLines[x][4][2] == 0 and y_1 == 1 and y_2 == 1 and y_3 == 0:
		MisclassOfSetosaForVersicolor += 1
	elif IrisTestDataLines[x][4][0] == 0 and IrisTestDataLines[x][4][1] == 0 and IrisTestDataLines[x][4][2] == 0 and y_1 == 1 and y_2 == 1 and y_3 == 1:
		MisclassOfSetosaForVerginica += 1
	
	if IrisTestDataLines[x][4][0] == 1 and IrisTestDataLines[x][4][1] == 1 and IrisTestDataLines[x][4][2] == 0 and y_1 == 0 and y_2 == 0 and y_3 == 0:
		MisclassOfVersicolorForSetosa += 1
	elif IrisTestDataLines[x][4][0] == 1 and IrisTestDataLines[x][4][1] == 1 and IrisTestDataLines[x][4][2] == 0 and y_1 == 1 and y_2 == 1 and y_3 == 1:
		MisclassOfVersicolorForVerginica += 1
		
	if IrisTestDataLines[x][4][0] == 1 and IrisTestDataLines[x][4][1] == 1 and IrisTestDataLines[x][4][2] == 1 and y_1 == 0 and y_2 == 0 and y_3 == 0:
		MisclassOfVerginicaForSetosa += 1
	elif IrisTestDataLines[x][4][0] == 1 and IrisTestDataLines[x][4][1] == 1 and IrisTestDataLines[x][4][2] == 1 and y_1 == 1 and y_2 == 1 and y_3 == 0:
		MisclassOfVerginicaForVersicolor += 1	
	
#diagonal of confusion matrix	
CorrectSetosa = CorrectOutputClassSetosa / 30
CorrectVersicolor = CorrectOutputClassVersicolor / 30
CorrectVerginica = CorrectOutputClassVirginica / 30

#remainder of the interior of the confusion matrix
SetosaMisclassVersicolor = MisclassOfSetosaForVersicolor / 30
SetosaMisclassVerginica = MisclassOfSetosaForVerginica / 30
VersicolorMisclassSetosa = MisclassOfVersicolorForSetosa / 30
VersicolorMisclassVerginica = MisclassOfVersicolorForVerginica / 30
VerginicaMisclassSetosa = MisclassOfVerginicaForSetosa / 30
VerginicaMisclassVersicolor = MisclassOfVerginicaForVersicolor / 30

#precision (right-most column of confusion matrix)
PrecisionSetosa = CorrectOutputClassSetosa / (CorrectOutputClassSetosa + MisclassOfVersicolorForSetosa + MisclassOfVerginicaForSetosa)
PrecisionVersicolor = CorrectOutputClassVersicolor / (CorrectOutputClassVersicolor + MisclassOfSetosaForVersicolor + MisclassOfVerginicaForVersicolor)
PrecisionVerginica = CorrectOutputClassVirginica / (CorrectOutputClassVirginica + MisclassOfSetosaForVerginica + MisclassOfVersicolorForVerginica)

#recal (bottom-most row of decision matrix)
RecallSetosa = CorrectOutputClassSetosa / (CorrectOutputClassSetosa + MisclassOfSetosaForVersicolor + MisclassOfSetosaForVerginica)
RecallVersicolor = CorrectOutputClassVersicolor / (CorrectOutputClassVersicolor + MisclassOfVersicolorForSetosa + MisclassOfVersicolorForVerginica)
RecallVerginica = CorrectOutputClassVirginica / (CorrectOutputClassVirginica + MisclassOfVerginicaForSetosa + MisclassOfVerginicaForVersicolor)

#overall accuracy (right-most and bottom-most corner of confusion matrix)
OverallAccuracy = (CorrectOutputClassSetosa + CorrectOutputClassVersicolor + CorrectOutputClassVirginica) / 30

print(CorrectSetosa, CorrectVerginica, CorrectVersicolor)
print(SetosaMisclassVersicolor, SetosaMisclassVersicolor, VersicolorMisclassSetosa, VersicolorMisclassVerginica, VerginicaMisclassSetosa, VerginicaMisclassVersicolor)
print(PrecisionSetosa, PrecisionVersicolor, PrecisionVerginica)
print(RecallSetosa, RecallVersicolor, RecallVerginica)
print(OverallAccuracy)
	