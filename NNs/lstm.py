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
import helping_functions


#--------------------------------------- RNN with LSTM layer


def LSTM(trainT, trainT_1, testT, testT_1, time_step, scaler, dataset):
	
	#create the model

	#we choose the Sequential because we want to stack the layers, put them in a row
	model = keras.models.Sequential()

	#we add a LSTM layer 
	#---> with 4 neurons or units
	#---> determine the input dimension based on the time_step because the input is going to be our previous values and the output will be only the predicted values
	#---> dropout: choose the percent to drop of the linear transformation of the reccurent state
	#---> implementation: choose if you want to stack the operation into larger number of smaller dot productes or the inverse
	#---> recurrent_dropout: the dropout of the recurrent state

	model.add(keras.layers.LSTM(128, input_shape=(1, time_step), use_bias=True, unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,return_sequences=True))

	#another one LSTM layer
	model.add(keras.layers.LSTM(64, input_shape=(1, time_step), return_sequences=False))

	model.add(keras.layers.Dense(16,init='uniform',activation='relu'))

	#just a densenly connected layer with 1 neuron/unit, as an output, that makes the single value prediction
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	#we use the RSME to validate the performance of our model, and the Adam optimizer for updating the network weights

	#Optimizers to use

	#----> Stochastic Gradient Descent - SGD
	#----> RMSProp 
	#----> Adagrad
	#----> Adam
	model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001))

	#feed our model
	results = model.fit(trainT, trainT_1, epochs=300, batch_size=1, verbose=1,validation_data=(testT, testT_1))


	#------------------ Make the predictions
	train_predict = model.predict(trainT)
	test_predict = model.predict(testT)

	#inverse the prediction in order to suit the format euros per time moment, for calculating the RMSE
	train_predict = scaler.inverse_transform(train_predict)
	trainT_1 = scaler.inverse_transform([trainT_1])
	test_predict = scaler.inverse_transform(test_predict)
	testT_1 = scaler.inverse_transform([testT_1])

	#now we can calculate the RMSE	

	train_score = math.sqrt(mean_squared_error(trainT_1[0], train_predict[:,0]))
	print('RMSE training: %.2f' % (train_score))

	test_score = math.sqrt(mean_squared_error(testT_1[0], test_predict[:,0]))
	print('RMSE testing: %.2f'% (test_score))

	Visualize(train_predict, test_predict, dataset, time_step, scaler, results)



#-------------- Visualize the pridictions
def Visualize(train_predict, test_predict, dataset, time_step, scaler, results):
	
	#initialize the array for testing and training
	train_predict_plot = np.empty_like(dataset)
	train_predict_plot[:, :] = np.nan

	test_predict_plot = np.empty_like(dataset)
	test_predict_plot[:, :] = np.nan


	#we have to shift the train predictions in order to plot them correctly
	train_predict_plot[time_step:len(train_predict)+time_step, :] = train_predict

	#we have to shift the test predictions in order to plot them correctly
	test_predict_plot[len(train_predict)+(time_step*2)+1:len(dataset)-1,:] = test_predict


	#plot baseline and the predictions in the same plot
	plt.figure(1)
	plt.title('Predictions from training and testins sets')
	plt.plot(scaler.inverse_transform(dataset))
	print(train_predict_plot)
	print(test_predict_plot)
	plt.plot(train_predict_plot)
	plt.plot(test_predict_plot)
	plt.legend(['Dataset','Train phase prediction','Test phase prediction'])

	plt.figure(2)
	plt.title('Train loss curve')
	plt.plot(results.history['loss'])
	plt.plot(results.history['val_loss'])
	plt.legend(['train loss','test loss'])

	#show the plots
	plt.show()
	


#------------------- Main procedure

def main():
	dataset = helping_functions.initialization()
	trainT, trainT_1, testT, testT_1, time_step, scaler, dataset = helping_functions.preprocessing(dataset)
	LSTM(trainT, trainT_1, testT, testT_1, time_step, scaler, dataset)


main()
