import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist

#load data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

#open saved matrix files
with open('w_2_1.txt', 'r') as f:
	w_2_1 = np.asarray([[float(num) for num in line.split(' ')] for line in f])

with open('w_1_0.txt', 'r') as r:
	w_1_0 = np.asarray([[float(num) for num in line.split(' ')] for line in r])
	
#function used to calculate output of nodes using sigmoidal activation function
def sigmoid(x):
	return 1/(1+np.exp(-x))
"""	
#ASSESS ON TRAINING DATA
	
#initialize vector to store prediction of trianing data
train_prediction = list()
	
for p in range(60000):
	#flatten data - convert from 28x28 matrix to 1x784 vector
	data_flattened = train_data[p].flatten()
	
	#bias absorbed so input 785 = 1
	data_bias_absorbed = np.append(data_flattened, 1)

	#compute hidden layers inputs for all 32 hidden nodes. Hidden node input is weighted sum of all input connections (use w(1,0) + train_data inputs)
	hidden_net_input_1 = np.sum([a*b for a,b in zip(w_1_0[:,0], data_bias_absorbed)])
	hidden_net_input_2 = np.sum([a*b for a,b in zip(w_1_0[:,1], data_bias_absorbed)])
	hidden_net_input_3 = np.sum([a*b for a,b in zip(w_1_0[:,2], data_bias_absorbed)])
	hidden_net_input_4 = np.sum([a*b for a,b in zip(w_1_0[:,3], data_bias_absorbed)])
	hidden_net_input_5 = np.sum([a*b for a,b in zip(w_1_0[:,4], data_bias_absorbed)])
	hidden_net_input_6 = np.sum([a*b for a,b in zip(w_1_0[:,5], data_bias_absorbed)])
	hidden_net_input_7 = np.sum([a*b for a,b in zip(w_1_0[:,6], data_bias_absorbed)])
	hidden_net_input_8 = np.sum([a*b for a,b in zip(w_1_0[:,7], data_bias_absorbed)])
	hidden_net_input_9 = np.sum([a*b for a,b in zip(w_1_0[:,8], data_bias_absorbed)])
	hidden_net_input_10 = np.sum([a*b for a,b in zip(w_1_0[:,9], data_bias_absorbed)])
	hidden_net_input_11 = np.sum([a*b for a,b in zip(w_1_0[:,10], data_bias_absorbed)])
	hidden_net_input_12 = np.sum([a*b for a,b in zip(w_1_0[:,11], data_bias_absorbed)])
	hidden_net_input_13 = np.sum([a*b for a,b in zip(w_1_0[:,12], data_bias_absorbed)])
	hidden_net_input_14 = np.sum([a*b for a,b in zip(w_1_0[:,13], data_bias_absorbed)])
	hidden_net_input_15 = np.sum([a*b for a,b in zip(w_1_0[:,14], data_bias_absorbed)])
	hidden_net_input_16 = np.sum([a*b for a,b in zip(w_1_0[:,15], data_bias_absorbed)])
	hidden_net_input_17 = np.sum([a*b for a,b in zip(w_1_0[:,16], data_bias_absorbed)])
	hidden_net_input_18 = np.sum([a*b for a,b in zip(w_1_0[:,17], data_bias_absorbed)])
	hidden_net_input_19 = np.sum([a*b for a,b in zip(w_1_0[:,18], data_bias_absorbed)])
	hidden_net_input_20 = np.sum([a*b for a,b in zip(w_1_0[:,19], data_bias_absorbed)])
	hidden_net_input_21 = np.sum([a*b for a,b in zip(w_1_0[:,20], data_bias_absorbed)])
	hidden_net_input_22 = np.sum([a*b for a,b in zip(w_1_0[:,21], data_bias_absorbed)])
	hidden_net_input_23 = np.sum([a*b for a,b in zip(w_1_0[:,22], data_bias_absorbed)])
	hidden_net_input_24 = np.sum([a*b for a,b in zip(w_1_0[:,23], data_bias_absorbed)])
	hidden_net_input_25 = np.sum([a*b for a,b in zip(w_1_0[:,24], data_bias_absorbed)])
	hidden_net_input_26 = np.sum([a*b for a,b in zip(w_1_0[:,25], data_bias_absorbed)])
	hidden_net_input_27 = np.sum([a*b for a,b in zip(w_1_0[:,26], data_bias_absorbed)])
	hidden_net_input_28 = np.sum([a*b for a,b in zip(w_1_0[:,27], data_bias_absorbed)])
	hidden_net_input_29 = np.sum([a*b for a,b in zip(w_1_0[:,28], data_bias_absorbed)])
	hidden_net_input_30 = np.sum([a*b for a,b in zip(w_1_0[:,29], data_bias_absorbed)])
	hidden_net_input_31 = np.sum([a*b for a,b in zip(w_1_0[:,30], data_bias_absorbed)])
	hidden_net_input_32 = np.sum([a*b for a,b in zip(w_1_0[:,31], data_bias_absorbed)])
	
	#combine hidden_net_inputs and create a vector (for debugging purposes)
	hidden_net_input_vector = [hidden_net_input_1, hidden_net_input_2, hidden_net_input_3, hidden_net_input_4, hidden_net_input_5, hidden_net_input_6, hidden_net_input_7, hidden_net_input_8, hidden_net_input_9, hidden_net_input_10, hidden_net_input_11, hidden_net_input_12, hidden_net_input_13, hidden_net_input_14, hidden_net_input_15, hidden_net_input_16, hidden_net_input_17, hidden_net_input_18, hidden_net_input_19, hidden_net_input_20, hidden_net_input_21, hidden_net_input_22, hidden_net_input_23, hidden_net_input_24, hidden_net_input_25, hidden_net_input_26, hidden_net_input_27, hidden_net_input_28, hidden_net_input_29, hidden_net_input_30, hidden_net_input_31, hidden_net_input_32]
	
	#compute hidden node outputs using sigmoidal activation function.
	hidden_output_1 = sigmoid(hidden_net_input_1)
	hidden_output_2 = sigmoid(hidden_net_input_2)
	hidden_output_3 = sigmoid(hidden_net_input_3)
	hidden_output_4 = sigmoid(hidden_net_input_4)
	hidden_output_5 = sigmoid(hidden_net_input_5)
	hidden_output_6 = sigmoid(hidden_net_input_6)
	hidden_output_7 = sigmoid(hidden_net_input_7)
	hidden_output_8 = sigmoid(hidden_net_input_8)
	hidden_output_9 = sigmoid(hidden_net_input_9)
	hidden_output_10 = sigmoid(hidden_net_input_10)
	hidden_output_11 = sigmoid(hidden_net_input_11)
	hidden_output_12 = sigmoid(hidden_net_input_12)
	hidden_output_13 = sigmoid(hidden_net_input_13)
	hidden_output_14 = sigmoid(hidden_net_input_14)
	hidden_output_15 = sigmoid(hidden_net_input_15)
	hidden_output_16 = sigmoid(hidden_net_input_16)
	hidden_output_17 = sigmoid(hidden_net_input_17)
	hidden_output_18 = sigmoid(hidden_net_input_18)
	hidden_output_19 = sigmoid(hidden_net_input_19)
	hidden_output_20 = sigmoid(hidden_net_input_20)
	hidden_output_21 = sigmoid(hidden_net_input_21)
	hidden_output_22 = sigmoid(hidden_net_input_22)
	hidden_output_23 = sigmoid(hidden_net_input_23)
	hidden_output_24 = sigmoid(hidden_net_input_24)
	hidden_output_25 = sigmoid(hidden_net_input_25)
	hidden_output_26 = sigmoid(hidden_net_input_26)
	hidden_output_27 = sigmoid(hidden_net_input_27)
	hidden_output_28 = sigmoid(hidden_net_input_28)
	hidden_output_29 = sigmoid(hidden_net_input_29)
	hidden_output_30 = sigmoid(hidden_net_input_30)
	hidden_output_31 = sigmoid(hidden_net_input_31)
	hidden_output_32 = sigmoid(hidden_net_input_32)

	#save outputs of hidden layer to array to be used in calculating net input for output nodes
	#bias absorbed so append a value of 1 to hidden_output_vector
	hidden_output_vector = [hidden_output_1, hidden_output_2, hidden_output_3, hidden_output_4, hidden_output_5, hidden_output_6, hidden_output_7, hidden_output_8, hidden_output_9, hidden_output_10, hidden_output_11, hidden_output_12, hidden_output_13, hidden_output_14, hidden_output_15, hidden_output_16, hidden_output_17, hidden_output_18, hidden_output_19, hidden_output_20, hidden_output_21, hidden_output_22, hidden_output_23, hidden_output_24, hidden_output_25, hidden_output_26, hidden_output_27, hidden_output_28, hidden_output_29, hidden_output_30, hidden_output_31, hidden_output_32, 1]

	#compute output layer inputs for all 10 output nodes. Output node input is the weighted sum of all hidden node outputs that connect to that output node
	output_net_input_1 = np.sum([a*b for a,b in zip(w_2_1[:,0], hidden_output_vector)])
	output_net_input_2 = np.sum([a*b for a,b in zip(w_2_1[:,1], hidden_output_vector)])
	output_net_input_3 = np.sum([a*b for a,b in zip(w_2_1[:,2], hidden_output_vector)])
	output_net_input_4 = np.sum([a*b for a,b in zip(w_2_1[:,3], hidden_output_vector)])
	output_net_input_5 = np.sum([a*b for a,b in zip(w_2_1[:,4], hidden_output_vector)])
	output_net_input_6 = np.sum([a*b for a,b in zip(w_2_1[:,5], hidden_output_vector)])
	output_net_input_7 = np.sum([a*b for a,b in zip(w_2_1[:,6], hidden_output_vector)])
	output_net_input_8 = np.sum([a*b for a,b in zip(w_2_1[:,7], hidden_output_vector)])
	output_net_input_9 = np.sum([a*b for a,b in zip(w_2_1[:,8], hidden_output_vector)])
	output_net_input_10 = np.sum([a*b for a,b in zip(w_2_1[:,9], hidden_output_vector)])
	
	#combine output_net_inputs and create a vector (for debugging purposes)
	output_net_input_vector = [output_net_input_1, output_net_input_2, output_net_input_3, output_net_input_4, output_net_input_5, output_net_input_6, output_net_input_7, output_net_input_8, output_net_input_9, output_net_input_10]
	
	#compute output layer outputs using sigmoidal activation function
	output_output_1 = sigmoid(output_net_input_1)
	output_output_2 = sigmoid(output_net_input_2)
	output_output_3 = sigmoid(output_net_input_3)
	output_output_4 = sigmoid(output_net_input_4)
	output_output_5 = sigmoid(output_net_input_5)
	output_output_6 = sigmoid(output_net_input_6)
	output_output_7 = sigmoid(output_net_input_7)
	output_output_8 = sigmoid(output_net_input_8)
	output_output_9 = sigmoid(output_net_input_9)
	output_output_10 = sigmoid(output_net_input_10)
	
	#save outputs of output layer to array to be used in error calculation
	output_output_vector = [output_output_1, output_output_2, output_output_3, output_output_4, output_output_5, output_output_6, output_output_7, output_output_8,output_output_9, output_output_10]
	
	print(p)
	
	train_prediction.append(np.argmax(output_output_vector))
	
#confusion matrix for training data
cmatrix_train = confusion_matrix(train_label, train_prediction)
print(cmatrix_train)
"""	
#ASSES ON TEST DATA

#initialize vector to store predictions of test data
test_prediction = list()

for p in range(10000):
	#flatten data - convert from 28x28 matrix to 1x784 vector
	data_flattened = test_data[p].flatten()
	
	#bias absorbed so input 785 = 1
	data_bias_absorbed = np.append(data_flattened, 1)

	#compute hidden layers inputs for all 32 hidden nodes. Hidden node input is weighted sum of all input connections (use w(1,0) + train_data inputs)
	hidden_net_input_1 = np.sum([a*b for a,b in zip(w_1_0[:,0], data_bias_absorbed)])
	hidden_net_input_2 = np.sum([a*b for a,b in zip(w_1_0[:,1], data_bias_absorbed)])
	hidden_net_input_3 = np.sum([a*b for a,b in zip(w_1_0[:,2], data_bias_absorbed)])
	hidden_net_input_4 = np.sum([a*b for a,b in zip(w_1_0[:,3], data_bias_absorbed)])
	hidden_net_input_5 = np.sum([a*b for a,b in zip(w_1_0[:,4], data_bias_absorbed)])
	hidden_net_input_6 = np.sum([a*b for a,b in zip(w_1_0[:,5], data_bias_absorbed)])
	hidden_net_input_7 = np.sum([a*b for a,b in zip(w_1_0[:,6], data_bias_absorbed)])
	hidden_net_input_8 = np.sum([a*b for a,b in zip(w_1_0[:,7], data_bias_absorbed)])
	hidden_net_input_9 = np.sum([a*b for a,b in zip(w_1_0[:,8], data_bias_absorbed)])
	hidden_net_input_10 = np.sum([a*b for a,b in zip(w_1_0[:,9], data_bias_absorbed)])
	hidden_net_input_11 = np.sum([a*b for a,b in zip(w_1_0[:,10], data_bias_absorbed)])
	hidden_net_input_12 = np.sum([a*b for a,b in zip(w_1_0[:,11], data_bias_absorbed)])
	hidden_net_input_13 = np.sum([a*b for a,b in zip(w_1_0[:,12], data_bias_absorbed)])
	hidden_net_input_14 = np.sum([a*b for a,b in zip(w_1_0[:,13], data_bias_absorbed)])
	hidden_net_input_15 = np.sum([a*b for a,b in zip(w_1_0[:,14], data_bias_absorbed)])
	hidden_net_input_16 = np.sum([a*b for a,b in zip(w_1_0[:,15], data_bias_absorbed)])
	hidden_net_input_17 = np.sum([a*b for a,b in zip(w_1_0[:,16], data_bias_absorbed)])
	hidden_net_input_18 = np.sum([a*b for a,b in zip(w_1_0[:,17], data_bias_absorbed)])
	hidden_net_input_19 = np.sum([a*b for a,b in zip(w_1_0[:,18], data_bias_absorbed)])
	hidden_net_input_20 = np.sum([a*b for a,b in zip(w_1_0[:,19], data_bias_absorbed)])
	hidden_net_input_21 = np.sum([a*b for a,b in zip(w_1_0[:,20], data_bias_absorbed)])
	hidden_net_input_22 = np.sum([a*b for a,b in zip(w_1_0[:,21], data_bias_absorbed)])
	hidden_net_input_23 = np.sum([a*b for a,b in zip(w_1_0[:,22], data_bias_absorbed)])
	hidden_net_input_24 = np.sum([a*b for a,b in zip(w_1_0[:,23], data_bias_absorbed)])
	hidden_net_input_25 = np.sum([a*b for a,b in zip(w_1_0[:,24], data_bias_absorbed)])
	hidden_net_input_26 = np.sum([a*b for a,b in zip(w_1_0[:,25], data_bias_absorbed)])
	hidden_net_input_27 = np.sum([a*b for a,b in zip(w_1_0[:,26], data_bias_absorbed)])
	hidden_net_input_28 = np.sum([a*b for a,b in zip(w_1_0[:,27], data_bias_absorbed)])
	hidden_net_input_29 = np.sum([a*b for a,b in zip(w_1_0[:,28], data_bias_absorbed)])
	hidden_net_input_30 = np.sum([a*b for a,b in zip(w_1_0[:,29], data_bias_absorbed)])
	hidden_net_input_31 = np.sum([a*b for a,b in zip(w_1_0[:,30], data_bias_absorbed)])
	hidden_net_input_32 = np.sum([a*b for a,b in zip(w_1_0[:,31], data_bias_absorbed)])
	
	#combine hidden_net_inputs and create a vector (for debugging purposes)
	hidden_net_input_vector = [hidden_net_input_1, hidden_net_input_2, hidden_net_input_3, hidden_net_input_4, hidden_net_input_5, hidden_net_input_6, hidden_net_input_7, hidden_net_input_8, hidden_net_input_9, hidden_net_input_10, hidden_net_input_11, hidden_net_input_12, hidden_net_input_13, hidden_net_input_14, hidden_net_input_15, hidden_net_input_16, hidden_net_input_17, hidden_net_input_18, hidden_net_input_19, hidden_net_input_20, hidden_net_input_21, hidden_net_input_22, hidden_net_input_23, hidden_net_input_24, hidden_net_input_25, hidden_net_input_26, hidden_net_input_27, hidden_net_input_28, hidden_net_input_29, hidden_net_input_30, hidden_net_input_31, hidden_net_input_32]
	
	#compute hidden node outputs using sigmoidal activation function.
	hidden_output_1 = sigmoid(hidden_net_input_1)
	hidden_output_2 = sigmoid(hidden_net_input_2)
	hidden_output_3 = sigmoid(hidden_net_input_3)
	hidden_output_4 = sigmoid(hidden_net_input_4)
	hidden_output_5 = sigmoid(hidden_net_input_5)
	hidden_output_6 = sigmoid(hidden_net_input_6)
	hidden_output_7 = sigmoid(hidden_net_input_7)
	hidden_output_8 = sigmoid(hidden_net_input_8)
	hidden_output_9 = sigmoid(hidden_net_input_9)
	hidden_output_10 = sigmoid(hidden_net_input_10)
	hidden_output_11 = sigmoid(hidden_net_input_11)
	hidden_output_12 = sigmoid(hidden_net_input_12)
	hidden_output_13 = sigmoid(hidden_net_input_13)
	hidden_output_14 = sigmoid(hidden_net_input_14)
	hidden_output_15 = sigmoid(hidden_net_input_15)
	hidden_output_16 = sigmoid(hidden_net_input_16)
	hidden_output_17 = sigmoid(hidden_net_input_17)
	hidden_output_18 = sigmoid(hidden_net_input_18)
	hidden_output_19 = sigmoid(hidden_net_input_19)
	hidden_output_20 = sigmoid(hidden_net_input_20)
	hidden_output_21 = sigmoid(hidden_net_input_21)
	hidden_output_22 = sigmoid(hidden_net_input_22)
	hidden_output_23 = sigmoid(hidden_net_input_23)
	hidden_output_24 = sigmoid(hidden_net_input_24)
	hidden_output_25 = sigmoid(hidden_net_input_25)
	hidden_output_26 = sigmoid(hidden_net_input_26)
	hidden_output_27 = sigmoid(hidden_net_input_27)
	hidden_output_28 = sigmoid(hidden_net_input_28)
	hidden_output_29 = sigmoid(hidden_net_input_29)
	hidden_output_30 = sigmoid(hidden_net_input_30)
	hidden_output_31 = sigmoid(hidden_net_input_31)
	hidden_output_32 = sigmoid(hidden_net_input_32)

	#save outputs of hidden layer to array to be used in calculating net input for output nodes
	#bias absorbed so append a value of 1 to hidden_output_vector
	hidden_output_vector = [hidden_output_1, hidden_output_2, hidden_output_3, hidden_output_4, hidden_output_5, hidden_output_6, hidden_output_7, hidden_output_8, hidden_output_9, hidden_output_10, hidden_output_11, hidden_output_12, hidden_output_13, hidden_output_14, hidden_output_15, hidden_output_16, hidden_output_17, hidden_output_18, hidden_output_19, hidden_output_20, hidden_output_21, hidden_output_22, hidden_output_23, hidden_output_24, hidden_output_25, hidden_output_26, hidden_output_27, hidden_output_28, hidden_output_29, hidden_output_30, hidden_output_31, hidden_output_32, 1]

	#compute output layer inputs for all 10 output nodes. Output node input is the weighted sum of all hidden node outputs that connect to that output node
	output_net_input_1 = np.sum([a*b for a,b in zip(w_2_1[:,0], hidden_output_vector)])
	output_net_input_2 = np.sum([a*b for a,b in zip(w_2_1[:,1], hidden_output_vector)])
	output_net_input_3 = np.sum([a*b for a,b in zip(w_2_1[:,2], hidden_output_vector)])
	output_net_input_4 = np.sum([a*b for a,b in zip(w_2_1[:,3], hidden_output_vector)])
	output_net_input_5 = np.sum([a*b for a,b in zip(w_2_1[:,4], hidden_output_vector)])
	output_net_input_6 = np.sum([a*b for a,b in zip(w_2_1[:,5], hidden_output_vector)])
	output_net_input_7 = np.sum([a*b for a,b in zip(w_2_1[:,6], hidden_output_vector)])
	output_net_input_8 = np.sum([a*b for a,b in zip(w_2_1[:,7], hidden_output_vector)])
	output_net_input_9 = np.sum([a*b for a,b in zip(w_2_1[:,8], hidden_output_vector)])
	output_net_input_10 = np.sum([a*b for a,b in zip(w_2_1[:,9], hidden_output_vector)])
	
	#combine output_net_inputs and create a vector (for debugging purposes)
	output_net_input_vector = [output_net_input_1, output_net_input_2, output_net_input_3, output_net_input_4, output_net_input_5, output_net_input_6, output_net_input_7, output_net_input_8, output_net_input_9, output_net_input_10]
	
	#compute output layer outputs using sigmoidal activation function
	output_output_1 = sigmoid(output_net_input_1)
	output_output_2 = sigmoid(output_net_input_2)
	output_output_3 = sigmoid(output_net_input_3)
	output_output_4 = sigmoid(output_net_input_4)
	output_output_5 = sigmoid(output_net_input_5)
	output_output_6 = sigmoid(output_net_input_6)
	output_output_7 = sigmoid(output_net_input_7)
	output_output_8 = sigmoid(output_net_input_8)
	output_output_9 = sigmoid(output_net_input_9)
	output_output_10 = sigmoid(output_net_input_10)
	
	#save outputs of output layer to array to be used in error calculation
	output_output_vector = [output_output_1, output_output_2, output_output_3, output_output_4, output_output_5, output_output_6, output_output_7, output_output_8,output_output_9, output_output_10]
	
	test_prediction.append(np.argmax(output_output_vector))

#confusion matrix for training data
cmatrix_test = confusion_matrix(test_label, test_prediction)
print(cmatrix_test)