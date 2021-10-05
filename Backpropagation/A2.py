#imports needed libraries
import numpy as np
import math
from tensorflow.keras.datasets import mnist

#load data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

#TRAINING
#create 2 weight matrices. One from input to hidden layer, the other for hidden to output layer
#w(1,0) is 785x32 (i x j matrix) where i=785 is the bias. w(2,1) is 33x10 (j x k matrix) where j=33 is the bias
#made this decision so when I calculate the error for a hidden node, the j index in w(1,0) will match the j index in w(2,1)
w_1_0 = np.random.uniform(low = -1, high = 1, size = (785,32))
w_2_1 = np.random.uniform(low = -1, high = 1, size = (33,10))

#initialize learning rate; random value chosen between 0 and 0.9
c = 0.5

#initialize alpha - momentum parameter
alpha = 0.5

#initialize epoch to zero; epoch to be used to limit the number of training iterations on entire input data set
epoch = 0

#one-hot encoded output values corresponding to 10 classes (10 digits from 0 to 9)
#use d(p,j) = 1 - e and d(p,l) = e instead of 1 and 0
#e = 0
digit_0 = [1,0,0,0,0,0,0,0,0,0]
digit_1 = [0,1,0,0,0,0,0,0,0,0]
digit_2 = [0,0,1,0,0,0,0,0,0,0]
digit_3 = [0,0,0,1,0,0,0,0,0,0]
digit_4 = [0,0,0,0,1,0,0,0,0,0]
digit_5 = [0,0,0,0,0,1,0,0,0,0]
digit_6 = [0,0,0,0,0,0,1,0,0,0]
digit_7 = [0,0,0,0,0,0,0,1,0,0]
digit_8 = [0,0,0,0,0,0,0,0,1,0]
digit_9 = [0,0,0,0,0,0,0,0,0,1]

#function used to calculate output of nodes using sigmoidal activation function
def sigmoid(x):
	return 1/(1+np.exp(-x))
	
#initialize error value
Error = 1000

#initialize 2 weight modification matrices that store weight modifications for each connection of the previous iteration. This is used to calculate momentum
w_modification_1_0 = np.zeros((785,32))
w_modification_2_1 = np.zeros((33,10))

#select max epoch value + minimum error before termination of training 
while(epoch < 3 and Error > 100):

	#initialize error sum variable to store total error of all 60,000 data points per epoch
	error_sum = 0
	
	#initialize data point number to zero (first data point)
	data_point_p = 0
	
	#iterate through all 60000 training data points
	while(data_point_p < 60000):
		
		#initialize count variable for per-batch training
		batch_count = 0
		
		#initialize empty matrix to store weight changes to be applied to original weight matrix after mini-batch
		w_2_1_batch = np.zeros((33,10))
		w_1_0_batch = np.zeros((785,32))
		
		while(batch_count < 250):
	
			#flatten data - convert from 28x28 matrix to 1x784 vector
			data_flattened = train_data[data_point_p].flatten()
			
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
			output_output_vector = [output_output_1, output_output_2, output_output_3, output_output_4, output_output_5, output_output_6, output_output_7, output_output_8, output_output_9, output_output_10]
			
			#convert ground truth label of point p to a k-length vector, where the value at index k represents the output node output at index k
			#use one-hot encoded arrays that represent the ground truth label digit of data point p
			if (train_label[data_point_p] == 0):
				truth_vector = digit_0
			elif (train_label[data_point_p] == 1):
				truth_vector = digit_1
			elif (train_label[data_point_p] == 2):
				truth_vector = digit_2
			elif (train_label[data_point_p] == 3):
				truth_vector = digit_3
			elif (train_label[data_point_p] == 4):
				truth_vector = digit_4
			elif (train_label[data_point_p] == 5):
				truth_vector = digit_5
			elif (train_label[data_point_p] == 6):
				truth_vector = digit_6
			elif (train_label[data_point_p] == 7):
				truth_vector = digit_7
			elif (train_label[data_point_p] == 8):
				truth_vector = digit_8
			elif (train_label[data_point_p] == 9):
				truth_vector = digit_9

			#initialize empty error vector
			error_vector = [0,0,0,0,0,0,0,0,0,0]
			
			#compute error vector where error at index k represents there error of class k
			for k in range(10):
				#take absolute value. To minimize error, we need error to move towards zero
				error_vector[k] = abs(output_output_vector[k] - truth_vector[k])

			#initialize empty delta vector to hold delta values (error) for each output node k
			delta_vector = [0,0,0,0,0,0,0,0,0,0]
			
			#calculate delta values for each output node k and store in delta_vector
			for x in range(10):
				delta_vector[x] = ((truth_vector[x]-output_output_vector[x])*output_output_vector[x]*(1-output_output_vector[x]))	

			#modify weight values for connections from hidden to output layer
			for j in range(33):
				for k in range(10):
					#calculate weight modification
					delta_w = ((c*delta_vector[k]*hidden_output_vector[j]) + (alpha*w_modification_2_1[j][k]))
					
					#store weight modification in per-batch matrix
					w_2_1_batch[j][k] = w_2_1_batch[j][k] + delta_w
					
					#store weight modification value in weight modification array to be used in momentum calculation for next iteration
					w_modification_2_1[j][k] = delta_w
					
			#initialize empty mu vector to store mu for all 32 hidden layer nodes
			mu_vector = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					
			#calculate mu for all 32 hidden nodes and store in vector of length 32
			for j in range(32):
				sum = 0
				#calculate weighted some of delta values for a hidden node j
				for k in range(10):
					sum = sum + (w_2_1[j][k]*delta_vector[k])
				
				#calculate mu for a hidden node j
				mu_vector[j] = sum*hidden_output_vector[j]*(1-hidden_output_vector[j])
			
			#modify weight values for connections from input to hidden layer
			for i in range(785):
				for j in range(32):
					#calculate weight modification
					delta_w = ((c*mu_vector[j]*data_bias_absorbed[i]) + (alpha*w_modification_1_0[i][j]))
					
					#store weight modification in per-batch matrix
					w_1_0_batch[i][j] = w_1_0_batch[i][j] + delta_w
					
					#store weight modification value in weight modification array to be used in momentum calculation for next iteration
					w_modification_1_0[i][j] = delta_w
			
			#sum errors of all p data points to be used to determine whether or not total error is acceptable
			error_sum = error_sum + np.sum(error_vector)

			#increment batch_count variable
			batch_count += 1
			
			#move to next data point
			data_point_p += 1

		#apply weight modifications to weight matrices after each mini-batch
		for j in range(33):
			for k in range(10):
				w_2_1[j][k] = w_2_1[j][k] + w_2_1_batch[j][k]
				
		for i in range(785):
			for j in range(32):
				w_1_0[i][j] = w_1_0[i][j] + w_1_0_batch[i][j]

	#store Error value for this epoch
	Error = error_sum


	#increment epoch
	epoch += 1

#store weight matrices in txt file
np.savetxt('w_2_1.txt', w_2_1)
np.savetxt('w_1_0.txt', w_1_0)
