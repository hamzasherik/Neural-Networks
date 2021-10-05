import keras
import sklearn
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#TRAINING
#load data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

#flatten data - convert 28X28 image data matrices to a 1x784 vector
train_data = train_data.reshape(train_data.shape[0], 28*28)
test_data = test_data.reshape(test_data.shape[0], 28*28)

#convert labels to one-hot encoded vectors
train_label = keras.utils.to_categorical(train_label, 10)
test_label = keras.utils.to_categorical(test_label, 10)

#specify sequential model where input layers feed hidden, and hidden feeds output
model = Sequential()

#create input to hidden layer
model.add(Dense(units=32, activation = 'sigmoid', input_shape = (784,)))

#create hidden to output layer
model.add(Dense(units=10, activation = 'sigmoid'))

#specify loss function and optimization function
model.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])

#train model
history = model.fit(train_data, train_label, epochs = 3, batch_size = 250)

#predict label on training data
label_prediction = model.predict(train_data)

#convert prediction to label_prediction
prediction = list()
for i in range(len(label_prediction)):
	prediction.append(np.argmax(label_prediction[i]))
	
#convert one-hot encoded train label to actual label
train = list()
for i in range(len(train_label)):
	train.append(np.argmax(train_label[i]))
	
#confusion matrix for training data
cmatrix_train = confusion_matrix(train, prediction)
print(cmatrix_train)

#TESTING
#check model performance on test data
label_prediction = model.predict(test_data)

#convert prediction to label_prediction
prediction = list()
for i in range(len(label_prediction)):
	prediction.append(np.argmax(label_prediction[i]))

#convert one-hot encoded test label to actual label
test = list()
for i in range(len(test_label)):
	test.append(np.argmax(test_label[i]))

#confusion matrix for test data
cmatrix_test = confusion_matrix(test, prediction)
print(cmatrix_test)
