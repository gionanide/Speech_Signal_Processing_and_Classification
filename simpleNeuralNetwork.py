#!usr/bin/python

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation , MaxPool2D , Conv2D , Flatten
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold

def readFeaturesFile():
	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
'Feature10','Feature11','Feature12','Feature13','Gender']
	data = pd.read_csv("mfcc_featuresLR.txt",names=names )
	#the outcome is a list of lists containing the samples with the following format
	#[charachteristic,feature1,feature2.......,feature13]
	#characheristic based on what we want for classification , can be (male , female) , also can be (normal-female,edema-female)
	#in general characheristic is the target value .
	return data

def preparingData(data):
	# Split-out validation dataset
	array = data.values
	#input
	X = array[:,0:13]
	#target 
	Y = array[:,13]
	
	#determine the test and the training size
	x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.10)
	

	#x_train = 
	#Encode the labels
	#reconstruct the data as a vector with a sequence of 1s and 0s
	y_train = keras.utils.to_categorical(y_train, num_classes = 2)
	y_test = keras.utils.to_categorical(y_test, num_classes = 2)
	'''print y_train.shape
	print x_train.shape
	print(y_train[0], np.argmax(y_train[0]))
	print(y_train[1], np.argmax(y_train[1]))
	print(y_train[2], np.argmax(y_train[2]))
	print(y_train[3], np.argmax(y_train[3]))'''
	return x_train , x_test , y_train , y_test 


def returnData(data):
	# Split-out validation dataset
	array = data.values
	#input
	X = array[:,0:13]
	#target 
	Y = array[:,13]
	
	#determine the test and the training size
	
	return X,Y



#Multilayer Perceptron
def testing_NN(data):
	X,Y = returnData(data)
	
	#determine the validation
	kfold = StratifiedKFold(n_splits=10,shuffle=True)
	#keep the results
	cvscores = []
	for train,test in kfold.split(X,Y):
		#Define a siple Multilayer Perceptron
		model = Sequential()

		#our classification is binary

		#as a first step we have to define the input dimensionality


		model.add(Dense(14,activation='relu',input_dim=13))


		#model.add(Dense(14,activation='relu',input_dim=13))
		model.add(Dense(8, activation='relu'))

		#add another hidden layer
		#model.add(Dense(16,activation='relu'))
		#the last step , add an output layer (number of neurons = number of classes)
		model.add(Dense(1,activation='sigmoid'))

		#select the optimizer
		#adam = Adam(lr=0.0001)
		adam = Adam(lr=0.001)
		#learning rate is between 0.0001 and 0.001 , but it is objective to define it
		#because we need out model not to learn to fast and maybe we have overfitting but also 
		#not to slow and take to much time . We can check this with the learning rate curve
	
		#we select the loss function and metrics that should be monitored
		#and then we compile our model
		model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

		#now we train our model
		model.fit(X[train],Y[train],epochs=50,batch_size=75,verbose=0)

			# evaluate the model
		scores = model.evaluate(X[test], Y[test], verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)


	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


	'''x_train , x_test , y_train , y_test = preparingData(data)
	
	#validation data =  test data
	#only for the plot
	results = model.fit(x_train,y_train,epochs=50,batch_size=75,verbose=2,validation_data=(x_test,y_test))

	plt.figure(1)
	plt.plot(results.history['loss'])
	plt.plot(results.history['val_loss'])
	plt.legend(['train loss', 'test loss'])
	plt.show()




	#now we can evaluate our model
	print '\n'
	print 'Train accuracy: ' , model.evaluate(x_train,y_train,batch_size=25)
	print 'Test accuracy: ',model.evaluate(x_test,y_test,batch_size=25)

	#visualize the actual output of the network
	output = model.predict(x_train)
	print '\n'
	print 'Actual output: ',output[0],np.argmax(output[0])

	#we can also check our model behaviour in depth
	print'\n'
	#print the first ten predictions
	for x in range(10):
		print 'Prediction: ',np.argsort(output[x])[::-1],'True target: ',np.argmax(y_train[x])'''

#Multilayer Perceptron
def simpleNN(data):
	x_train , x_test , y_train , y_test = preparingData(data)

	#because as we can see from the previous function simpleNN the
	#test loss is going bigger which means that we have overfitting problem
	#here we are going to try to overcome this obstacle

	model = Sequential()

	#The input layer:
	'''With respect to the number of neurons comprising this layer, this parameter is completely and uniquely determined 
	once you know the shape of your training data. Specifically, the number of neurons comprising that layer is equal to the number 
	of features (columns) in your data. Some NN configurations add one additional node for a bias term.'''

	model.add(Dense(14,activation='relu',input_dim=13,kernel_initializer='random_uniform'))
	
	#The output layer
	'''If the NN is a classifier, then it also has a single node unless
	 softmax is used in which case the output layer has one node per 
	class label in your model.'''
	
	model.add(Dense(2,activation='softmax'))

	#binary_crossentropy because we have a binary classification model
	#Because it is not guaranteed that we are going to find the global optimum
	#because we can be trapped in a local minima and the algorithm may think that
	#you reach global minima. To avoid this situation, we use a momentum term in the 
	#objective function, which is a value 0 < momentum < 1 , that increases the size of the steps
	#taken towards the minimum by trying to jump from a local minima.


	#If the momentum term is large then the learning rate should be kept smaller.
	#A large value of momentum means that the convergence will happen fast,but if
	#both are kept at large values , then we might skip the minimum with a huge step.
	#A small value of momentum cannot reliably avoid local minima, and also slow down
	#the training system. We are trying to find the right value of momentum through cross-validation.	
	model.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.001,momentum=0.6),metrics=['accuracy'])

	#In simple terms , learning rate is how quickly a network abandons old beliefs for new ones.
	#Which means that with a higher LR the network changes its mind more quickly , in pur case this means
	#how quickly our model update the parameters (weights,bias).

	#verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

	results = model.fit(x_train,y_train,epochs=50,batch_size=50,verbose=2,validation_data=(x_test,y_test))

	print 'Train accuracy: ' , model.evaluate(x_train,y_train,batch_size=50,verbose=2)
	print 'Test accuracy: ',model.evaluate(x_test,y_test,batch_size=50,verbose=2)


	
	#visualize
	plt.figure(1)
	plt.plot(results.history['loss'])
	plt.plot(results.history['val_loss'])
	plt.legend(['train loss', 'test loss'])
	plt.show()

	print model.summary()



def main():
	data = readFeaturesFile()
	simpleNN(data)
		

main()
