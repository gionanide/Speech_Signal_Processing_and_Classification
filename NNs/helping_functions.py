#!usr/bin/python

#libraries
import pandas as pd
import matplotlib.pyplot as plt	
import numpy as np
import math
import keras.models
import keras.layers 
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
import keras.optimizers


'''First we have to define our problem. We use LSTM because we want to take advandage of it' strong point which is Long-Short term memory, contrary to RNN which we know there are drawbacks concerning the gradient in the first layers, so maybe we have a leakage of previous information. For long time series or long sequencies in general we use LSTMs, GRUs bu for small sequencies RNN  are efficient as well.'''


#------------------- Initialization


#make this function for reading the input data and make the format you want, and visualize 
def initialization():

	print('Remember for small datasets emerges problem with zero elements in the testing set, if the training percent is a big number. \n ')


	#import some randomness in our procedure
	np.random.seed(7)

	#determine the path of the file that you want to read from
	-----------------------------------------------------------------> Define the path
	path = ''

	#print the two destinations that consit the trip of, split the string based on '/' and print the last word subtracking the last four charachters '.txt'
	print('Running for: ',path.split('/')[6][:-4])

	#define the name of the DataFrame columns
	names=["x","y"]

	#read the file that we define to the path as a pandas DataFrame with the aforementioned columns, target is the ticket_price, we can feed our model with the usecols
	timeserie = pd.read_csv(path, names = names,engine='python', index_col=None, usecols = ["x","y"]
	#visualize the DataFrame, and check the dimensions
	print('Dataframe: \n')
	print(timeserie)
	print(timeserie.shape)
	print('\n')

	#plot the how the price is evolving through time (definition of time day, month, year etc)
	#plt.plot(timeserie)
	#plt.show()


	#returns the file as a Dataframe
	return timeserie





#------------------ Split Data training/testing

#split data into training and testing subsets
def split_data(dataset, training_size):
	
	#translate the training size into our number of elements
	train_size = int(len(dataset) * training_size)
	test_size = len(dataset) - train_size

	#take the accordinate parts of the datraset
	train_samples, test_samples = dataset[0:train_size,:], dataset[train_size: len(dataset),:]

	return train_samples, test_samples



#------------------------------------- 	Format Dataset


'''We are making this function because we want to change the format of our data, we are going to implement regression so to predict the next value of a timeserie. This means that we are goint to have a specific time, let's say t, and we are predicting what is happening in the time t+1, so we need to model our dataset in order to implement this ideology'''
def format_dataset(dataset, time_step):
	
	#time_step defines how many times you want to look back

	#define as dataT the time t, and as dataT_1 the time t+1
	dataT, dataT_1 = [], []

	#iterate all the dataset and make the format based on the time step
	for i in range(len(dataset)-time_step-1):

		#in time t put the current element
		dataT.append(dataset[i:(i+time_step), 0])

		#in the time t+1 put the next element of the element that we append in the list dataT
		dataT_1.append(dataset[i + time_step, 0])

		#repeat this procedure, following the element that we append in the array dataT_1
	
	
	return np.array(dataT), np.array(dataT_1)



#---------------- Preprocessing


#we use this function in order to do the preprocessing staff, normalize, and maybe another procedures #that we want to implement for making a proper format to our data
def preprocessing(dataset, time_step, training_size):
	
	#take only tha information of the dataframe and not the indexes or the columns names
	dataset = dataset.values

	#convert them to floats which is more suitable for feeding a neural netowork
	dataset = dataset.astype('float32')

	#first we are going to scale our data because LSTMs are sensitive to the unscaled input data and we are goint to see this in action
	#scaling in range [0,1]
	scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))


	#----------------------------------------------------------------> fit all the dataset

	dataset = scaler.fit_transform(dataset)

	#and then split
	train_samples_scaled, test_samples_scaled = split_data(dataset)

	#----------------------------------------------------------------> fit only training

	#first we have to split our data into test and training
	train_samples, test_samples = split_data(dataset, training_size)

	print('Samples chosen for the training procedure: \n')
	print(train_samples)
	print(train_samples.shape)
	print('\n')
	print('Samples chose for testing: \n')
	print(test_samples)
	print(test_samples.shape)
	print('\n')

	'''
	scaler.fit(train_samples)

	#transform both training and testing data based on the information of the training only because we want our model to work only for one sample for testing as input
	train_samples_scaled = scaler.transform(train_samples)

	test_samples_scaled = scaler.transform(test_samples)'''

	#visualize the scaled data
	print('Scaled samples for the training procedure: \n')
	print(train_samples_scaled)
	print(train_samples_scaled.shape)
	print('\n')
	print('Scaled samples for testing: \n')
	print(test_samples_scaled)
	print(test_samples_scaled.shape)
	print('\n')

	#------------------------------------------------------------------> end with scaling

	#timeserie format for both testing and training sets

	#define the time step, e.g t+5 we want time_step=5	
	time_step=1

	trainT, trainT_1 = format_dataset(train_samples_scaled, time_step)

	testT, testT_1 = format_dataset(test_samples_scaled, time_step)

	print('Previous train data shape: ')
	print(trainT)
	print('\n')
	
	#bacause the LSTM waits our input to be in the format below
	#[samples, time steps, features]
	#we need to transform it in order to fit this prerequisite

	#------------------------------------------------------------> [samples, time steps, features] format
	#we are formating only the training set not the testing, because the testing in going to be just  apredicted single valuew
	trainT = np.reshape(trainT, (trainT.shape[0], 1, trainT.shape[1]))

	testT = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

	print('Current train data shape ready to feed LSTM model: ')
	print(trainT)
	print('\n')

	
	#return the sets of training and testing ready for the LSTM model
	return trainT, trainT_1, testT, testT_1, time_step, scaler, dataset

